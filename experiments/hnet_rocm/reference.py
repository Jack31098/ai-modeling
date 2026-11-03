"""Full batch-one H-Net reference. No custom CUDA/Triton kernels or CPU fallbacks.

Preserves all checkpoint parameter names and the original recursive architecture.
Uses the already checked encoder equations plus explicit STE and dechunk SSD.
Not the upstream fused implementation; cache/packed batches are not implemented.
"""

from .bootstrap import RUNTIME

import json
from pathlib import Path
import re

import torch
from torch import nn
import torch.nn.functional as F

from experiments.hnet.inspect_hierarchy import EncoderProbe, rms, ssd_blocks, ssd_serial


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probability):
        return torch.ones_like(probability)

    @staticmethod
    def backward(ctx, gradient):
        return gradient


class TrackedWeights(dict):
    def __init__(self, pairs, used):
        super().__init__(pairs)
        self.used = used

    def __getitem__(self, key):
        self.used.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self:
            self.used.add(key)
        return super().get(key, default)


def dechunk(hidden, selected, probability, length, mode="official-bf16-scan"):
    """Original EMA scan equations, including forward's explicit BF16 conversions.

    dt=-log(1-p); x=hidden/dt; A=-1; B=p; C=1. The original forward
    rounds dt, x and B to BF16; scan accumulates FP32 and returns BF16.
    ema-fp32-step instead matches the mathematical recurrence in DeChunkLayer.step.
    """
    p = probability[selected].float().clamp(1e-4, 1 - 1e-4)
    state = hidden.new_zeros(hidden.shape[-1], dtype=torch.float32)
    outputs = []
    if mode == "official-bf16-scan":
        dt = (-torch.log1p(-p)).to(torch.bfloat16)
        x = (hidden / dt[:, None]).to(torch.bfloat16)
        b = p.to(torch.bfloat16)
        for i in range(len(hidden)):
            state = (-dt[i].float()).exp() * state + dt[i].float() * x[i].float() * b[i].float()
            outputs.append(state.to(torch.bfloat16).to(hidden.dtype))
    elif mode == "ema-fp32-step":
        for i in range(len(hidden)):
            state = p[i] * hidden[i].float() + (1 - p[i]) * state
            outputs.append(state.to(hidden.dtype))
    else:
        raise ValueError("unknown dechunk mode")
    coarse = torch.stack(outputs)
    mask = torch.zeros(length, device=hidden.device, dtype=torch.long).scatter(0, selected, 1)
    index = mask.cumsum(0) - 1
    return coarse[index]


class ReferenceHNet(nn.Module):
    # Reuse unchanged, independently checked Mamba2/SDPA encoder equations.
    linear = EncoderProbe.linear
    mamba = EncoderProbe.mamba
    attention = EncoderProbe.attention

    def __init__(self, checkpoint, config, device="cpu", scan="blocks", dechunk_mode="official-bf16-scan"):
        super().__init__()
        self.config = config
        self.scan = ssd_blocks if scan == "blocks" else ssd_serial
        self.dechunk_mode = dechunk_mode
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if not all(isinstance(v, torch.Tensor) and v.dtype == torch.float32 for v in state.values()):
            raise ValueError("expected original FP32 tensor-only checkpoint")
        for name, tensor in state.items():
            module = self
            parts = name.split(".")
            for part in parts[:-1]:
                if part not in module._modules:
                    module.add_module(part, nn.Module())
                module = module._modules[part]
            # Clone CPU mmap too: optimizer steps must never modify the checkpoint file.
            value = tensor.to(device=device).clone()
            module.register_parameter(parts[-1], nn.Parameter(value))
        if set(self.state_dict()) != set(state):
            raise ValueError("checkpoint key coverage failure")
        self.used = set()
        self.w = TrackedWeights(self.named_parameters(), self.used)
        self.execution_trace = []

    def isotropic(self, x, prefix, layout, stage):
        residual, index = None, 0
        for arch, count in re.findall(r"([mT])(\d+)", layout):
            for _ in range(int(count)):
                block = f"{prefix}.layers.{index}"
                residual = x if residual is None else x + residual
                h = rms(residual, self.w[block + ".norm1.weight"])
                x = self.mamba(h, block + ".mixer") if arch == "m" else self.attention(h, block + ".mixer", stage)
                if arch == "T":
                    residual = x + residual
                    h = rms(residual, self.w[block + ".norm2.weight"])
                    value, gate = self.linear(h, block + ".mlp.fc1").chunk(2, -1)
                    x = self.linear(value * F.silu(gate), block + ".mlp.fc2")
                index += 1
        result = rms(x + residual, self.w[prefix + ".rmsnorm.weight"])
        self.execution_trace.append({"module": prefix, "positions": len(result), "device": str(result.device), "dtype": str(result.dtype)})
        return result

    def hierarchy(self, x, prefix, layout, stage, routes):
        original_width = x.shape[-1]
        pad_key = prefix + ".pad_dimension"
        if pad_key in self.w:
            x = torch.cat((x, self.w[pad_key].expand(len(x), -1)), -1)
        if len(layout) == 1:
            return self.isotropic(x, prefix + ".main_network", layout[0], stage)[:, :original_width]
        h = self.isotropic(x, prefix + ".encoder", layout[0], stage)
        residual = self.linear(h.float(), prefix + ".residual_proj")
        router = prefix + ".routing_module"
        q = F.normalize(self.linear(h[:-1], router + ".q_proj_layer"), dim=-1)
        k = F.normalize(self.linear(h[1:], router + ".k_proj_layer"), dim=-1)
        probability = F.pad(((1 - (q * k).sum(-1)) / 2).clamp(0, 1), (1, 0), value=1)
        mask = probability > .5  # argmax([1-p,p]) chooses 0 at exactly .5.
        selected = torch.where(mask)[0]
        chosen_probability = torch.where(mask, probability, 1 - probability)
        routes.append({"stage": stage, "probability": probability, "selected": selected,
                       "selected_probability": chosen_probability})
        inner = self.hierarchy(h[selected], prefix + ".main_network", layout[1], stage + 1, routes)
        expanded = dechunk(inner, selected, probability, len(h), self.dechunk_mode)
        h = expanded.float() * STE.apply(chosen_probability[:, None]) + residual
        result = self.isotropic(h, prefix + ".decoder", layout[2], stage)
        return result[:, :original_width]

    def forward(self, ids):
        if ids.ndim != 1 or not ids.numel():
            raise ValueError("reference supports one nonempty unpadded byte sequence only")
        if ids.device != self.w["embeddings.weight"].device:
            raise ValueError("input/parameter device mismatch; no silent fallback")
        self.execution_trace = []
        self.used.clear()
        hidden = F.embedding(ids, self.w["embeddings.weight"])
        routes = []
        hidden = self.hierarchy(hidden, "backbone", self.config["arch_layout"], 0, routes)
        logits = self.linear(hidden, "lm_head")
        return logits, routes

    def unused_parameters(self):
        return sorted(set(self.w) - self.used)


def load_config(path="models/hnet_2stage_L/config.json"):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

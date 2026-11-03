"""Inspect original H-Net's two learned routers with the official FP32 weights.

Portable encoder-only reference: PyTorch SDPA and Mamba-2 SSD equations replace
CUDA/Triton kernels. No decoder, innermost LM, sampling, or training is run.
Source revisions and checkpoint hash are pinned in model-locks/hnet-2stage-l.json.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


def rms(x, weight, eps=1e-5):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight


def ssd_serial(x, dt, a, b, c):
    """Direct recurrence, equivalent to Mamba2.step's ngroups=1 fallback."""
    state = x.new_zeros(x.shape[1], x.shape[2], b.shape[-1])
    result = []
    for i in range(len(x)):
        state = (
            (dt[i] * a).exp()[:, None, None] * state
            + x[i, :, :, None] * (dt[i, :, None] * b[i])[..., None, :]
        )
        result.append(torch.einsum("hpn,n->hp", state, c[i]))
    return torch.stack(result)


def ssd_blocks(x, dt, a, b, c, chunk=32):
    """Blockwise semiseparable evaluation, with zero initial state.

    Within a block, y_i=sum_{j<=i}(C_i.B_j)*exp(sum_{k=j+1..i}dt_k*A)
    *dt_j*x_j. Carry state between blocks to avoid a quadratic full sequence.
    """
    state = x.new_zeros(x.shape[1], x.shape[2], b.shape[-1])
    result = []
    for start in range(0, len(x), chunk):
        xx, dd = x[start:start + chunk], dt[start:start + chunk]
        bb, cc = b[start:start + chunk], c[start:start + chunk]
        sums = (dd * a).T.cumsum(-1)
        length = len(xx)
        lower = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        log_decay = sums[:, :, None] - sums[:, None, :]
        decay = log_decay.masked_fill(~lower, -torch.inf).exp()
        cb = cc @ bb.T
        local = torch.bmm(decay * cb, (xx * dd[:, :, None]).permute(1, 0, 2))
        carry = torch.einsum("tn,hpn->thp", cc, state) * sums.T.exp()[:, :, None]
        result.append(local.permute(1, 0, 2) + carry)
        end_decay = (sums[:, -1:] - sums).exp() * dd.T
        state = (
            sums[:, -1].exp()[:, None, None] * state
            + torch.einsum("ht,thp,tn->hpn", end_decay, xx, bb)
        )
    return torch.cat(result)


class EncoderProbe:
    def __init__(self, checkpoint, config, device="cpu"):
        self.config = config
        self.source_dir = Path(checkpoint).parent / "source"
        self.device = torch.device(device)
        self.sd = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        assert all(isinstance(v, torch.Tensor) for v in self.sd.values())
        self.parameters = sum(v.numel() for v in self.sd.values())
        # These are all weights needed to form the two routers' inputs.
        prefixes = (
            "embeddings.", "backbone.encoder.", "backbone.routing_module.",
            "backbone.main_network.encoder.", "backbone.main_network.routing_module.",
        )
        self.w = {k: v.to(self.device) for k, v in self.sd.items() if k.startswith(prefixes)}
        self.used_parameters = sum(v.numel() for v in self.w.values())
        assert config["arch_layout"] == ["m4", ["T1m4", ["T26"], "m4T1"], "m4"]
        assert config["d_model"] == [1024, 1024, 1536]
        self.scan = ssd_blocks

    def linear(self, x, prefix):
        return F.linear(x, self.w[prefix + ".weight"], self.w.get(prefix + ".bias"))

    def mamba(self, x, prefix):
        cfg = self.config["ssm_cfg"]
        d_inner = cfg["expand"] * x.shape[-1]
        d_state = cfg["d_state"]
        nheads = self.w[prefix + ".dt_bias"].numel()
        head_dim = d_inner // nheads
        assert head_dim == 64
        projected = self.linear(x, prefix + ".in_proj")
        z, xbc, dt = projected.split([d_inner, d_inner + 2 * d_state, nheads], -1)
        conv = F.conv1d(
            xbc.T.unsqueeze(0), self.w[prefix + ".conv1d.weight"],
            self.w[prefix + ".conv1d.bias"], padding=cfg["d_conv"] - 1,
            groups=xbc.shape[-1],
        )[0, :, :len(x)].T
        xx, bb, cc = F.silu(conv).split([d_inner, d_state, d_state], -1)
        xx = xx.reshape(len(x), nheads, head_dim)
        dt = F.softplus(dt + self.w[prefix + ".dt_bias"])
        a = -self.w[prefix + ".A_log"].exp()
        y = self.scan(xx, dt, a, bb, cc)
        y = y + xx * self.w[prefix + ".D"][None, :, None]
        y = y.reshape(len(x), d_inner)
        # Mamba2 defaults: rmsnorm=True, norm_before_gate=False, ngroups=1.
        y = rms(y * F.silu(z), self.w[prefix + ".norm.weight"])
        return self.linear(y, prefix + ".out_proj")

    def attention(self, x, prefix, stage):
        length, width = x.shape
        heads = self.config["attn_cfg"]["num_heads"][stage]
        rot = self.config["attn_cfg"]["rotary_emb_dim"][stage]
        q, k, v = self.linear(x, prefix + ".Wqkv").reshape(length, 3, heads, -1).unbind(1)
        inv = 10000.0 ** (-torch.arange(0, rot, 2, device=x.device).float() / rot)
        angles = torch.outer(torch.arange(length, device=x.device).float(), inv)
        cos, sin = angles.cos()[:, None, :], angles.sin()[:, None, :]

        def rotate(t):
            first, second = t[..., :rot].chunk(2, -1)
            return torch.cat((first * cos - second * sin, second * cos + first * sin,
                              t[..., rot:]), -1)

        q, k = rotate(q), rotate(k)
        q, k, v = (t.permute(1, 0, 2).unsqueeze(0) for t in (q, k, v))
        window = self.config["attn_cfg"]["window_size"][stage]
        if window < 0 or length <= window + 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            offsets = torch.arange(length, device=x.device)
            delta = offsets[:, None] - offsets[None, :]
            mask = (delta >= 0) & (delta <= window)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.linear(y[0].permute(1, 0, 2).reshape(length, width), prefix + ".out_proj")

    def encoder(self, x, prefix, stage):
        arches = "mmmm" if stage == 0 else "Tmmmm"
        residual = None
        for i, arch in enumerate(arches):
            block = f"{prefix}.layers.{i}"
            residual = x if residual is None else x + residual
            h = rms(residual, self.w[block + ".norm1.weight"])
            x = self.mamba(h, block + ".mixer") if arch == "m" else self.attention(h, block + ".mixer", stage)
            if arch == "T":
                residual = x + residual
                h = rms(residual, self.w[block + ".norm2.weight"])
                value, gate = self.linear(h, block + ".mlp.fc1").chunk(2, -1)
                x = self.linear(value * F.silu(gate), block + ".mlp.fc2")
        return rms(x + residual, self.w[prefix + ".rmsnorm.weight"])

    def route(self, h, prefix):
        q = F.normalize(self.linear(h[:-1], prefix + ".q_proj_layer"), dim=-1)
        k = F.normalize(self.linear(h[1:], prefix + ".k_proj_layer"), dim=-1)
        p = F.pad(((1 - (q * k).sum(-1)) / 2).clamp(0, 1), (1, 0), value=1)
        selected = torch.where(p > 0.5)[0]
        assert selected[0].item() == 0
        return selected, p

    @torch.inference_mode()
    def inspect(self, data):
        assert data, "Empty input has no measurable node rate"
        ids = torch.tensor(list(data), dtype=torch.long, device=self.device)
        h0 = F.embedding(ids, self.w["embeddings.weight"])
        h0 = self.encoder(h0, "backbone.encoder", 0)
        first, p0 = self.route(h0, "backbone.routing_module")
        h1 = self.encoder(h0[first], "backbone.main_network.encoder", 1)
        second, p1 = self.route(h1, "backbone.main_network.routing_module")
        return first.cpu(), second.cpu(), p0.cpu(), p1.cpu()


def visible_spans(data, starts):
    ends = starts[1:] + [len(data)]
    return [
        {"start_byte": a, "end_byte": b, "length_bytes": b - a,
         "text": data[a:b].decode("utf-8", errors="backslashreplace"),
         "hex": data[a:b].hex()}
        for a, b in zip(starts, ends)
    ]


def measure(probe, name, text):
    data = text.encode("utf-8")
    start = time.perf_counter()
    i0, i1, p0, p1 = probe.inspect(data)
    mapped = i0[i1].tolist()
    n0, n1, n2 = len(data), len(i0), len(i1)
    result = {
        "name": name, "text": text, "characters": len(text),
        "whitespace_word_count": len(text.split()),
        "bytes": n0, "stage1_nodes": n1, "stage2_nodes": n2,
        "stage1_nodes_per_byte": n1 / n0,
        "stage2_nodes_per_byte": n2 / n0,
        "bytes_per_stage1_node": n0 / n1,
        "stage1_nodes_per_stage2_node": n1 / n2,
        "bytes_per_stage2_node": n0 / n2,
        "stage1_boundary_byte_offsets": i0.tolist(),
        "stage2_boundary_stage1_offsets": i1.tolist(),
        "stage2_boundary_byte_offsets": mapped,
        "stage1_boundary_probabilities": p0.tolist(),
        "stage2_boundary_probabilities": p1.tolist(),
        "stage1_near_threshold": int(((p0 - 0.5).abs() < 1e-3).sum()),
        "stage2_near_threshold": int(((p1 - 0.5).abs() < 1e-3).sum()),
        "stage1_spans": visible_spans(data, i0.tolist()),
        "stage2_spans": visible_spans(data, mapped),
        "seconds": time.perf_counter() - start,
    }
    assert sum(s["length_bytes"] for s in result["stage1_spans"]) == n0
    assert sum(s["length_bytes"] for s in result["stage2_spans"]) == n0
    return result


def verify(probe):
    generator = torch.Generator().manual_seed(928)
    x = torch.randn(73, 3, 4, generator=generator, dtype=torch.float64)
    dt = torch.rand(73, 3, generator=generator, dtype=torch.float64) * 0.1
    a = -torch.rand(3, generator=generator, dtype=torch.float64)
    b = torch.randn(73, 5, generator=generator, dtype=torch.float64)
    c = torch.randn(73, 5, generator=generator, dtype=torch.float64)
    expected = ssd_serial(x, dt, a, b, c)
    actual = ssd_blocks(x, dt, a, b, c)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)

    data = b"The student builds a smaller model from the previous complete model."
    fast = probe.inspect(data)
    probe.scan = ssd_serial
    slow = probe.inspect(data)
    probe.scan = ssd_blocks
    for i in (0, 1):
        assert torch.equal(fast[i], slow[i]), "Scan evaluation changed a boundary"
    for i in (2, 3):
        torch.testing.assert_close(fast[i], slow[i], atol=1e-4, rtol=1e-4)
    unusual_cases_checked = []
    for name, text in samples():
        if name not in ("chinese", "random_ascii", "repeat_letter"):
            continue
        fast_case = probe.inspect(text.encode("utf-8"))
        probe.scan = ssd_serial
        slow_case = probe.inspect(text.encode("utf-8"))
        probe.scan = ssd_blocks
        assert all(torch.equal(fast_case[i], slow_case[i]) for i in (0, 1))
        unusual_cases_checked.append(name)
    extended = probe.inspect(data + b" This sentence is an unseen future suffix.")
    assert torch.equal(fast[0], extended[0][extended[0] < len(data)])
    assert torch.equal(fast[1], extended[1][extended[1] < len(fast[0])])
    torch.testing.assert_close(fast[2], extended[2][:len(data)], atol=1e-4, rtol=1e-4)
    single = probe.inspect(b"x")
    assert single[0].tolist() == [0] and single[1].tolist() == [0]

    # Execute only the reviewed router classes from the pinned official source,
    # excluding the CUDA-dependent dechunking classes and all other imports.
    source = probe.source_dir / "hnet__modules__dc.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    official_classes = ast.Module(body=[node for node in tree.body if isinstance(node, ast.ClassDef)
        and node.name in {"RoutingModuleOutput", "RoutingModuleState", "RoutingModule"}], type_ignores=[])
    scope = {"torch": torch, "nn": torch.nn, "F": F, "dataclass": dataclass}
    exec(compile(official_classes, str(source), "exec"), scope)
    official_router = scope["RoutingModule"](1024).to(probe.device)
    official_router.load_state_dict({key.removeprefix("backbone.routing_module."): value
        for key, value in probe.w.items() if key.startswith("backbone.routing_module.")})
    with torch.inference_mode():
        hidden = F.embedding(torch.tensor(list(data), device=probe.device), probe.w["embeddings.weight"])
        hidden = probe.encoder(hidden, "backbone.encoder", 0)
        original = official_router(hidden.unsqueeze(0), mask=torch.ones(1, len(data), dtype=torch.bool, device=probe.device))
    assert torch.equal(torch.where(original.boundary_mask[0])[0].cpu(), fast[0])
    torch.testing.assert_close(original.boundary_prob[0, :, 1].cpu(), fast[2])
    return {
        "ssd_block_vs_serial_max_abs_error_fp64": float((actual - expected).abs().max()),
        "actual_weight_scan_methods_identical_boundaries": True,
        "actual_weight_router_probability_max_abs_difference":
            max(float((fast[i] - slow[i]).abs().max()) for i in (2, 3)),
        "future_suffix_prefix_boundaries_unchanged": True,
        "single_byte_forced_boundary": True,
        "official_router_source_matches": True,
        "unusual_cases_scan_methods_identical_boundaries": unusual_cases_checked,
        "official_cuda_kernel_parity": "not tested; this is a portable reference encoder probe",
    }


def samples():
    rng = random.Random(20261102)
    original = [
        ("english", "A hierarchical language model reads a stream of bytes and groups them into larger units. "
         "The first encoder handles local patterns, while the inner network processes a shorter sequence. "
         "We measure the number of selected positions at each stage, rather than assuming the target compression ratio was reached."),
        ("chinese", "我们想观察层级模型实际怎样压缩文本。第一层把字节序列变成较少的节点，第二层继续选择更粗的节点。"
         "边界数量减少并不自动证明语义抽象成功，还要检查人物、否定、数值和条件是否被保留下来。"),
        ("python", "def remaining_budget(total, expenses):\n    used = 0\n    for expense in expenses:\n"
         "        used += expense\n    return total - used\n\nassert remaining_budget(10, [3]) == 7\n"
         "assert remaining_budget(10, [7]) == 3\n"),
        ("repeat_letter", "T" * 320),
        ("repeat_phrase", "The budget is ten. " * 20),
        ("random_ascii", "".join(rng.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(320))),
        ("new_name_context", "Zorvellan is a new identifier. Zorvellan owns a blue box. "
         "The Zorvellan record has a budget. We update Zorvellan, then read Zorvellan again. "
         "Zorvellan returns three units. Zorvellan now has seven units left."),
    ]
    phrase = "The red box holds 10 coins.\n"
    count = 20
    total = len(phrase) * count
    numbers = rng.sample(range(10, 100), count)
    varied_numbers = "".join(f"The red box holds {n} coins.\n" for n in numbers)
    varied_fields = "".join(
        f"The {rng.choice(['red', 'tan', 'dry', 'wet'])} "
        f"{rng.choice(['box', 'bag', 'jar', 'cup'])} holds {n} "
        f"{rng.choice(['coins', 'seeds', 'beans', 'cards', 'books'])}.\n"
        for n in numbers
    )
    prose = (
        "A visitor found a small notebook beside the window. Its owner had drawn a map of the village, "
        "but one bridge was missing. Before leaving, the visitor asked a baker whether the river could be crossed. "
        "The baker pointed toward a narrow road and warned that rain had damaged the eastern bank. "
        "They checked the weather, packed some food, and chose a different route through the hills. "
        "At sunset they reached a quiet station, where a delayed train gave them time to compare their notes. "
        "The journey took longer than expected, yet the new route revealed an old orchard and a sheltered path. "
    )
    controlled = [
        ("redundancy_exact_repeat", phrase * count),
        ("redundancy_vary_numbers", varied_numbers),
        ("redundancy_vary_fields", varied_fields),
        ("redundancy_varied_prose", (prose * 2)[:total]),
        ("redundancy_random_ascii", "".join(rng.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(total))),
    ]
    assert all(len(text.encode('utf-8')) == 560 for _, text in controlled)
    lengths = [(f"repeat_length_{n}", phrase * n) for n in (1, 4, 16)]
    return original + controlled + lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("models/hnet_2stage_L"))
    parser.add_argument("--output", type=Path, default=Path("experiments/hnet/results/hierarchy.json"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8-sig"))
    probe = EncoderProbe(args.model_dir / "hnet_2stage_L.pt", config, args.device)
    checks = verify(probe)
    print(json.dumps({"checks": checks}, ensure_ascii=True), flush=True)
    results = []
    for name, text in samples():
        result = measure(probe, name, text)
        results.append(result)
        print(f"{name}: {result['bytes']} -> {result['stage1_nodes']} -> {result['stage2_nodes']}; "
              f"{result['bytes_per_stage2_node']:.3f} bytes/inner-node; {result['seconds']:.2f}s", flush=True)
    counts = [sum(s[k] for s in results) for k in ("bytes", "stage1_nodes", "stage2_nodes")]
    payload = {
        "model": "cartesia-ai/hnet_2stage_L",
        "revision": "eb0db7323b1131e6bdb8d1c7367d3f815d09426c",
        "parameters": probe.parameters, "encoder_probe_parameters": probe.used_parameters,
        "torch": torch.__version__, "device": str(probe.device), "dtype": "float32",
        "threads": args.threads, "checks": checks,
        "scope": "two encoders and routers only; no inner LM or decoders executed",
        "tokenization": "raw UTF-8 bytes; no BOS/EOS; state reset per sample",
        "boundary_semantics": "selected byte positions projected to raw input; original lookahead H-Net; displayed spans are not independently decoded semantic codes",
        "rate_semantics": "sequence node rates, not entropy-coded bits/byte or latent storage compression",
        "aggregate": {
            "bytes": counts[0], "stage1_nodes": counts[1], "stage2_nodes": counts[2],
            "bytes_per_stage1_node": counts[0] / counts[1],
            "stage1_nodes_per_stage2_node": counts[1] / counts[2],
            "bytes_per_stage2_node": counts[0] / counts[2],
        },
        "samples": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "aggregate": payload["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()

"""Reviewable HIP preflight, full-model forward/backward and CPU parity runs."""

from .bootstrap import RUNTIME

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import time
import traceback

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .reference import ReferenceHNet, STE, dechunk, load_config, rms, ssd_blocks, ssd_serial

BASE = Path(__file__).parent
MODEL = Path("models/hnet_2stage_L")
SAMPLE = "The red box holds 17 coins. Keep it by the window."
PARITY_ATOL, PARITY_RTOL = 1e-3, 1e-3


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(obj, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def checked_device(device):
    if device.startswith("cuda") and (not torch.version.hip or not torch.cuda.is_available()):
        raise RuntimeError("HIP device required; refusing CPU fallback")


def metadata():
    return {"client_date": "2026-11-03", "timezone": "America/Los_Angeles", "python": platform.python_version(),
            "torch": torch.__version__, "hip": torch.version.hip, "runtime": RUNTIME,
            "implementation_sha256": {p.name: sha(p) for p in sorted(BASE.glob("*.py"))},
            "reused_encoder_sha256": sha("experiments/hnet/inspect_hierarchy.py")}


def preflight():
    checked_device("cuda")
    props = torch.cuda.get_device_properties(0)
    result = {**metadata(), "device": torch.cuda.get_device_name(0), "architecture": props.gcnArchName,
              "total_memory_bytes": props.total_memory, "free_total_before": torch.cuda.mem_get_info(),
              "scope": "single-process HIP tensor and autograd tests; not general platform training support"}
    x = torch.arange(256, dtype=torch.float32).reshape(16, 16).to("cuda").requires_grad_()
    y = (x @ x.T).square().mean()
    y.backward()
    torch.cuda.synchronize()
    result["tensor_test"] = {"status": "passed", "tensor_device": str(x.device), "output_device": str(y.device),
                             "gradient_device": str(x.grad.device), "output_finite": bool(torch.isfinite(y)),
                             "gradient_finite": bool(torch.isfinite(x.grad).all()), "loss": y.detach().item()}
    result["custom_modules"] = {}
    for name in ("triton", "pytorch_triton_rocm", "mamba_ssm", "causal_conv1d", "flash_attn"):
        spec = importlib.util.find_spec(name)
        result["custom_modules"][name] = {"status": "not-installed" if spec is None else "importable-spec-only", "kernel_execution": "not-run"}
    result["status"] = "passed"
    return result


def parity_ops():
    """GPU FP32 forward/VJP vs CPU independent FP64 or explicit scalar formulas."""
    checked_device("cuda")
    generator = torch.Generator().manual_seed(20261103)
    rows = []

    def check(name, values, expected_fn, actual_fn, atol=2e-4, rtol=2e-4):
        cpu = [x.double().requires_grad_() for x in values]
        gpu = [x.float().to("cuda").requires_grad_() for x in values]
        expected, actual = expected_fn(*cpu), actual_fn(*gpu)
        weight = torch.linspace(.1, 1, expected.numel(), dtype=torch.float64).reshape(expected.shape)
        cg = torch.autograd.grad((expected * weight).sum(), cpu)
        gg = torch.autograd.grad((actual * weight.float().to("cuda")).sum(), gpu)
        torch.testing.assert_close(actual.cpu().double(), expected, atol=atol, rtol=rtol)
        errors = []
        for a, b in zip(gg, cg):
            torch.testing.assert_close(a.cpu().double(), b, atol=atol, rtol=rtol)
            errors.append((a.cpu().double() - b).abs().max().item())
        rows.append({"operator": name, "status": "passed", "forward_max_abs": (actual.cpu().double() - expected).abs().max().item(),
                     "vjp_max_abs": max(errors), "atol": atol, "rtol": rtol, "actual_device": str(actual.device)})

    x = torch.randn(19, 2, 3, generator=generator)
    dt = torch.rand(19, 2, generator=generator) * .1
    a = -torch.rand(2, generator=generator)
    b, c = (torch.randn(19, 5, generator=generator) for _ in range(2))
    check("Mamba2-SSD-blocks-vs-direct-recurrence", [x, dt, a, b, c], ssd_serial, ssd_blocks)
    x, w, bias = torch.randn(1, 5, 17, generator=generator), torch.randn(5, 1, 4, generator=generator), torch.randn(5, generator=generator)
    def explicit_conv(x, w, bias):
        windows = F.pad(x, (3, 0)).unfold(-1, 4, 1)
        return (windows * w[:, 0][None, :, None, :]).sum(-1) + bias[None, :, None]
    check("causal-depthwise-conv1d-vs-explicit-unfold", [x, w, bias], explicit_conv,
          lambda x, w, bias: F.conv1d(x, w, bias, padding=3, groups=5)[..., :x.shape[-1]])
    q, k, v = (torch.randn(1, 2, 9, 8, generator=generator) for _ in range(3))
    def explicit_attention(q, k, v):
        scores = q @ k.transpose(-1, -2) / q.shape[-1] ** .5
        allowed = torch.ones(9, 9, dtype=torch.bool, device=q.device).tril()
        return scores.masked_fill(~allowed, -torch.inf).softmax(-1) @ v
    check("SDPA-math-vs-explicit-causal-attention", [q, k, v], explicit_attention,
          lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True))
    x, w = torch.randn(7, 16, generator=generator), torch.randn(16, generator=generator)
    check("RMSNorm", [x, w], lambda x, w: x / (x.square().mean(-1, keepdim=True) + 1e-5).sqrt() * w, rms)
    hidden, p = torch.randn(4, 12, generator=generator), torch.tensor([.9999, .2, .8, .3, .1, .7, .2])
    selected_cpu, selected_gpu = torch.tensor([0, 2, 4, 6]), torch.tensor([0, 2, 4, 6], device="cuda")
    def scalar_ema(h, p):
        vals, state = [], h.new_zeros(h.shape[-1])
        pp = p[selected_cpu].clamp(1e-4, 1-1e-4)
        for i in range(len(h)):
            state = pp[i] * h[i] + (1-pp[i]) * state
            vals.append(state)
        return torch.stack(vals)[torch.tensor([0,0,1,1,2,2,3])]
    check("dechunk-EMA-step-vs-direct-recurrence", [hidden, p], scalar_ema,
          lambda h, p: dechunk(h, selected_gpu, p, 7, "ema-fp32-step"))
    z = torch.rand(7, device="cuda", requires_grad=True)
    value = STE.apply(z)
    value.sum().backward()
    assert torch.equal(value, torch.ones_like(value)) and torch.equal(z.grad, torch.ones_like(z))
    rows.append({"operator": "routing-STE", "status": "passed", "forward": "all ones", "gradient": "identity", "device": str(z.device)})
    return {**metadata(), "status": "passed", "tests": rows, "native_fused_parity": "not-run; custom fused modules absent"}


def route_metadata(routes):
    return [{"stage": r["stage"], "input_length": len(r["probability"]), "output_length": len(r["selected"]),
             "selected": r["selected"].detach().cpu().tolist(), "probability": r["probability"].detach().cpu().tolist(),
             "min_threshold_margin": (r["probability"].detach() - .5).abs().min().item(),
             "within_1e-3_of_threshold": int(((r["probability"].detach() - .5).abs() < 1e-3).sum())}
            for r in routes]


def full(device, train=False, scan="blocks", dechunk_mode="ema-fp32-step"):
    checked_device(device)
    torch.set_num_threads(8)
    torch.manual_seed(20261103)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = ReferenceHNet(MODEL / "hnet_2stage_L.pt", load_config(MODEL / "config.json"), device, scan, dechunk_mode)
    report = {**metadata(), "device": device, "dtype": "float32", "batch_size": 1, "training": train,
              "scope": "complete portable reference; original checkpoint, both routers, inner T26, both decoders, final logits",
              "parameter_count": sum(p.numel() for p in model.parameters()), "parameter_tensors": len(list(model.parameters())),
              "parameter_devices": sorted({str(p.device) for p in model.parameters()}),
              "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
              "load_seconds": time.perf_counter() - started, "scan": scan, "dechunk_mode": dechunk_mode,
              "model_lock": json.loads(Path("model-locks/hnet-2stage-l.json").read_text(encoding="utf-8")),
              "config_sha256": sha(MODEL / "config.json"), "config": model.config, "input_text": SAMPLE,
              "tokenization": "raw UTF-8 bytes; no BOS/EOS; loss predicts byte i+1 from bytes through i"}
    data = SAMPLE.encode("utf-8")
    ids = torch.tensor(list(data), device=device)
    started = time.perf_counter()
    if train:
        logits, routes = model(ids[:-1])
        loss = F.cross_entropy(logits, ids[1:])
    else:
        with torch.inference_mode():
            logits, routes = model(ids[:-1])
            loss = F.cross_entropy(logits, ids[1:])
    sync(device)
    report["forward"] = {"status": "passed", "seconds": time.perf_counter() - started,
                         "input_bytes": list(data[:-1]), "target_bytes": list(data[1:]),
                         "shape": list(logits.shape), "logits_finite": bool(torch.isfinite(logits).all()),
                         "loss_finite": bool(torch.isfinite(loss)), "next_byte_loss": loss.detach().item(),
                         "logits_device": str(logits.device), "logits": logits.detach().cpu().tolist(),
                         "routes": route_metadata(routes), "executed_modules": model.execution_trace,
                         "unused_checkpoint_keys": model.unused_parameters()}
    assert report["parameter_count"] == 874315264 and not model.unused_parameters()
    assert report["forward"]["logits_finite"] and report["forward"]["loss_finite"]
    report["backward"] = {"status": "not-run"}
    report["optimizer"] = {"status": "not-run"}
    if train:
        started = time.perf_counter()
        loss.backward()
        sync(device)
        elapsed = time.perf_counter() - started
        gradients = []
        for name, param in model.named_parameters():
            grad = param.grad
            gradients.append({"name": name, "numel": param.numel(), "present": grad is not None,
                              "finite": bool(torch.isfinite(grad).all()) if grad is not None else False,
                              "norm": grad.norm().item() if grad is not None else None,
                              "device": str(grad.device) if grad is not None else None})
        assert all(g["present"] and g["finite"] and g["device"] == str(logits.device) for g in gradients)
        report["backward"] = {"status": "passed", "seconds": elapsed, "frozen_parameters": 0,
                              "gradients": gradients, "zero_norm_names": [g["name"] for g in gradients if g["norm"] == 0]}
        names = ["lm_head.weight", "backbone.routing_module.q_proj_layer.weight",
                 "backbone.main_network.routing_module.q_proj_layer.weight",
                 "backbone.main_network.main_network.main_network.layers.0.mixer.Wqkv.weight",
                 "backbone.decoder.layers.0.mixer.in_proj.weight", "backbone.main_network.decoder.layers.0.mixer.in_proj.weight"]
        before = {name: model.get_parameter(name).detach().clone() for name in names}
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=.9, foreach=False)
        started = time.perf_counter()
        optimizer.step()
        sync(device)
        updates = {name: (model.get_parameter(name).detach() - old).abs().max().item() for name, old in before.items()}
        states = [v for state in optimizer.state.values() for v in state.values() if isinstance(v, torch.Tensor)]
        assert all(v > 0 for v in updates.values()) and all(torch.isfinite(v).all() and v.device == logits.device for v in states)
        report["optimizer"] = {"status": "passed", "name": "SGD", "lr": 1e-5, "momentum": .9,
                               "foreach": False, "steps": 1, "seconds": time.perf_counter() - started,
                               "sampled_parameter_max_abs_updates": updates, "state_tensors": len(states),
                               "state_numel": sum(v.numel() for v in states),
                               "state_devices": sorted({str(v.device) for v in states}), "state_all_finite": True,
                               "checkpoint_written": False}
    if device.startswith("cuda"):
        report["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["free_total_after"] = torch.cuda.mem_get_info()
    report["status"] = "passed"
    return report


def compare(cpu_path, gpu_path):
    cpu, gpu = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (cpu_path, gpu_path))
    assert cpu["device"] == "cpu" and gpu["device"].startswith("cuda")
    assert cpu["forward"]["input_bytes"] == gpu["forward"]["input_bytes"]
    assert cpu["model_lock"] == gpu["model_lock"] and cpu["dechunk_mode"] == gpu["dechunk_mode"]
    a, b = torch.tensor(cpu["forward"]["logits"]), torch.tensor(gpu["forward"]["logits"])
    difference = (a - b).abs()
    equal_routes = [x["selected"] == y["selected"] for x, y in zip(cpu["forward"]["routes"], gpu["forward"]["routes"])]
    probability_errors = [max(abs(a-b) for a,b in zip(x["probability"],y["probability"]))
                          if len(x["probability"]) == len(y["probability"]) else None
                          for x,y in zip(cpu["forward"]["routes"],gpu["forward"]["routes"])]
    close = torch.allclose(a, b, atol=PARITY_ATOL, rtol=PARITY_RTOL)
    return {**metadata(), "status": "passed" if close and all(equal_routes) else "failed",
            "cpu_report_sha256": sha(cpu_path), "gpu_report_sha256": sha(gpu_path), "atol": PARITY_ATOL, "rtol": PARITY_RTOL,
            "logits_max_abs": difference.max().item(), "logits_mean_abs": difference.mean().item(),
            "same_boundary_positions_by_stage": equal_routes, "probability_max_abs_by_stage": probability_errors,
            "next_byte_loss_difference": abs(cpu["forward"]["next_byte_loss"] - gpu["forward"]["next_byte_loss"]),
            "near_threshold": gpu["forward"]["routes"], "limitation": "same FP32 weights; CPU serial SSD versus HIP block SSD; does not validate unavailable original fused kernels"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "ops", "full", "compare"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--scan", choices=("serial", "blocks"), default="blocks")
    parser.add_argument("--dechunk-mode", choices=("ema-fp32-step", "official-bf16-scan"), default="ema-fp32-step")
    parser.add_argument("--cpu-report", type=Path)
    parser.add_argument("--gpu-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("choose a new report path")
    try:
        # Explicit math SDPA keeps the portable reference independent of optional AOTriton.
        with sdpa_kernel(SDPBackend.MATH):
            if args.phase == "preflight":
                report = preflight()
            elif args.phase == "ops":
                report = parity_ops()
            elif args.phase == "full":
                report = full(args.device, args.train, args.scan, args.dechunk_mode)
            else:
                report = compare(args.cpu_report, args.gpu_report)
    except Exception:
        report = {**metadata(), "status": "failed", "traceback": traceback.format_exc()}
    save(args.out, report)
    print(json.dumps({"phase": args.phase, "status": report["status"], "out": str(args.out)}), flush=True)
    if report["status"] != "passed":
        print(report.get("traceback", "comparison failed"), flush=True)
        raise SystemExit(1)

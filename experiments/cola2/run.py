"""Run the upstream quickstart and a short reconstruction probe on local assets."""

import argparse
import ast
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import struct
import sys
import time
import traceback

from .acquire import CODE_REV, LOCAL, MODEL_REV, ROOT, SHARDS, now

PROMPT = "Question: What is the capital of France? Answer:"
RECON_TEXT = "The red box holds 17 coins. Keep it by the window."
SOURCE = LOCAL / f"source-{CODE_REV[:12]}" / "src"
SNAPSHOT = LOCAL / f"hf-{MODEL_REV[:12]}"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def code_identity():
    evidence = {}
    for name in ("configuration_cola2.py", "modeling_cola2.py", "modeling_layers.py"):
        github = (SOURCE / "cola2" / name).read_bytes()
        hf = (SNAPSHOT / name).read_bytes()
        equal = ast.dump(ast.parse(github)) == ast.dump(ast.parse(hf))
        if name != "configuration_cola2.py" and not equal:
            raise ValueError("unexpected computation source difference")
        evidence[name] = {"github_sha256": hashlib.sha256(github).hexdigest(),
                          "hf_sha256": hashlib.sha256(hf).hexdigest(), "python_ast_equal": equal}
    return evidence


def inspect_checkpoint():
    """Validate index and logical tensor byte ranges without reading tensor bodies."""
    index_document = read_json(SNAPSHOT / "model.safetensors.index.json")
    index = index_document["weight_map"]
    lock = read_json(LOCAL / "assets.lock.json")
    declared = {x["id"]: x for x in lock["files"] if x["id"] in SHARDS}
    verified = {x["id"]: x for x in read_json(LOCAL / "verified.json")["shards"]}
    if set(verified) != SHARDS or set(index.values()) != SHARDS:
        raise ValueError("missing verified shard or unexpected index file")
    mapping, elements, buffer_elements = {}, Counter(), {}
    for name in sorted(SHARDS):
        path = SNAPSHOT / name
        size = path.stat().st_size
        if size != declared[name]["size_bytes"] or verified[name]["sha256"] != declared[name]["sha256"]:
            raise ValueError(f"inconsistent verification receipt for {name}")
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            if length > 2_000_000:
                raise ValueError("oversized tensor header")
            header = json.loads(stream.read(length))
        end = 0
        tensors = ((k, v) for k, v in header.items() if k != "__metadata__")
        for key, tensor in sorted(tensors, key=lambda item: item[1]["data_offsets"][0]):
            if key in mapping:
                raise ValueError(f"duplicate tensor {key}")
            mapping[key] = name
            count = math.prod(tensor["shape"])
            start, stop = tensor["data_offsets"]
            if tensor["dtype"] != "F32" or start != end or stop - start != count * 4:
                raise ValueError(f"unexpected shape, dtype or byte range: {key}")
            end = stop
            elements[".".join(key.split(".")[:2]) if key.startswith("vae.") else key.split(".")[0]] += count
            # Registered by upstream DynamicSegmenter, so it is stored but not a parameter.
            if key == "vae.segmenter.rate_bias":
                buffer_elements[key] = count
        if 8 + length + end != size:
            raise ValueError(f"shard size does not agree with tensor directory: {name}")
    total = sum(elements.values())
    nominal = lock["safetensors"]["total"]
    if (mapping != index or total * 4 != index_document["metadata"]["total_size"]
            or buffer_elements != {"vae.segmenter.rate_bias": 1} or total - sum(buffer_elements.values()) != nominal):
        raise ValueError(f"checkpoint index or parameter metadata disagreement: stored={total}, parameters={nominal}, buffers={buffer_elements}")
    return {"tensor_count": len(mapping), "stored_elements_by_module": dict(elements),
            "stored_elements": total, "hf_parameter_count": nominal, "persisted_buffer_elements": buffer_elements,
            "weight_file_bytes": lock["weights_bytes"], "tensor_payload_bytes": total * 4,
            "stored_dtype": "F32", "validated_shards": sorted(verified)}


def run(args, record, save):
    temporary = LOCAL / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ.update(TEMP=str(temporary), TMP=str(temporary), HF_HOME=str(LOCAL / "hf-cache"),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    sys.path[0:0] = [str(LOCAL / "deps"), str(SOURCE)]
    if args.device == "cuda" and os.name == "nt":
        from experiments.hnet_rocm.bootstrap import RUNTIME
        record["msvc_preload"] = {k: v for k, v in RUNTIME.items() if k != "path"}
    save("importing_torch")
    import torch
    import transformers
    import safetensors
    import accelerate
    import psutil
    from cola2 import Cola2Config, Cola2ForConditionalGeneration, Cola2GenerationConfig
    from transformers import AutoTokenizer
    from torch.nn.attention import sdpa_kernel, SDPBackend

    torch.set_num_threads(args.threads)
    process = psutil.Process()
    use_gpu = args.device == "cuda"
    device = torch.device("cuda:0" if use_gpu else "cpu")
    dtype = getattr(torch, args.dtype)

    def sync():
        if use_gpu:
            torch.cuda.synchronize()

    def memory():
        host = process.memory_info()
        data = {"process_working_set_bytes": host.rss, "process_peak_working_set_bytes": getattr(host, "peak_wset", None)}
        if use_gpu:
            data.update(torch_allocated_bytes=torch.cuda.memory_allocated(), torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        torch_reserved_bytes=torch.cuda.memory_reserved(), torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        hip_free_total_bytes=list(torch.cuda.mem_get_info()))
        return data

    record["environment"] = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                             "hip": torch.version.hip, "transformers": transformers.__version__,
                             "safetensors": safetensors.__version__, "accelerate": accelerate.__version__,
                             "cpu_threads": args.threads, "sdpa": "MATH" if use_gpu else "CPU default"}
    if use_gpu:
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError("requested ROCm GPU is unavailable; no implicit CPU fallback")
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is not supported on this device")
        props = torch.cuda.get_device_properties(device)
        record["environment"].update(gpu=props.name, gpu_arch=getattr(props, "gcnArchName", None),
                                     hip_total_bytes=props.total_memory)
        if args.tiny and dtype == torch.bfloat16:
            with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                a = torch.ones((32, 32), device=device, dtype=dtype)
                product = a @ a
                q = a.reshape(1, 1, 32, 32)
                attention = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
                sync()
                passed = (product.dtype == dtype and attention.dtype == dtype
                          and product.device == attention.device == device
                          and bool((product == 32).all()) and bool((attention == 1).all()))
                record["bf16_smoke"] = {"passed": passed, "operators": ["matmul", "causal SDPA MATH"],
                                         "device": str(device), "dtype": str(dtype)}
                if not passed:
                    raise RuntimeError("GPU BF16 smoke check disagreed with expected values")
                del a, product, q, attention
            torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    record["source_code_comparison"] = code_identity()
    record["memory_before_load"] = memory()
    save("loading")
    start = time.perf_counter()
    if args.tiny:
        torch.manual_seed(7)
        model = Cola2ForConditionalGeneration(Cola2Config.tiny()).to(device=device, dtype=dtype).eval()
        tokenizer = None
        inputs = {"input_ids": torch.tensor([[1, 11, 12, 13]], device=device)}
        options = {"generation_config": Cola2GenerationConfig(max_new_tokens=3, dit_num_steps=2,
                   dit_guidance_scale=2.0, decoder_guidance_scale=-0.9, eos_token_id=None,
                   pad_token_id=0, do_sample=False, dynamic_duration=True, latent_feedback=True)}
    else:
        record["checkpoint"] = inspect_checkpoint()
        tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
        load_options = {}
        if args.direct_device_load:
            # Both Cola2 submodels stay on the GPU. A non-root map lets the
            # safetensors loader cast each CPU slice before transfer to HBM.
            load_options["device_map"] = {name: str(device) for name in ("vae", "dit")}
        model, loading = Cola2ForConditionalGeneration.from_pretrained(
            SNAPSHOT, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True,
            output_loading_info=True, **load_options)
        record["loading_info"] = loading
        if any(loading.values()):
            raise ValueError(f"upstream load was not exact: {loading}")
        if not args.direct_device_load:
            model = model.to(device)
        model.eval()
        record["checkpoint_loaded"] = True
        inputs = dict(tokenizer(PROMPT, return_tensors="pt").to(device))
        options = {} if args.max_new_tokens is None else {"max_new_tokens": args.max_new_tokens}
    sync()
    record["load_wall_seconds"] = time.perf_counter() - start
    groups, dtypes, devices = Counter(), Counter(), Counter()
    for name, parameter in model.named_parameters():
        groups[".".join(name.split(".")[:2]) if name.startswith("vae.") else name.split(".")[0]] += parameter.numel()
        dtypes[str(parameter.dtype)] += parameter.numel()
        devices[str(parameter.device)] += parameter.numel()
    if set(devices) != {str(device)}:
        raise RuntimeError(f"unexpected parameter placement: {devices}")
    if set(dtypes) != {str(dtype)}:
        raise RuntimeError(f"unexpected parameter dtypes: {dtypes}")
    record["buffer_placement"] = dict(Counter(
        f"{buffer.device} {buffer.dtype}" for buffer in model.buffers()))
    record["load_device_map"] = getattr(model, "hf_device_map", None)
    record["cpu_offload_configured"] = False
    if not args.tiny:
        checkpoint = record["checkpoint"]
        buffers = dict(model.named_buffers())
        if (sum(groups.values()) != checkpoint["hf_parameter_count"]
                or any(buffers[k].numel() != n for k, n in checkpoint["persisted_buffer_elements"].items())):
            raise ValueError("runtime parameter/buffer count differs from inspected checkpoint")
    record.update(parameter_count=sum(groups.values()), parameters_by_module=dict(groups),
                  parameter_dtypes=dict(dtypes), parameter_devices=dict(devices), memory_after_load=memory())
    record["generation_config"] = model.generation_config.to_dict() if not args.tiny else options["generation_config"].to_dict()
    save("generating")
    if use_gpu:
        torch.cuda.reset_peak_memory_stats()
    context = sdpa_kernel(SDPBackend.MATH) if use_gpu else nullcontext()
    with torch.inference_mode(), context:
        start = time.perf_counter()
        generated = model.generate(**inputs, **options, generator=torch.Generator(device=device).manual_seed(args.seed))
        sync()
        elapsed = time.perf_counter() - start
        ids = inputs["input_ids"][0]
        if not torch.equal(generated[0, :ids.numel()], ids):
            raise ValueError("output does not preserve the input prefix")
        completion = generated[0, ids.numel():].tolist()
        if not completion:
            raise ValueError("no generated token")
        record["generation"] = {"prompt": None if args.tiny else PROMPT, "prompt_ids": ids.tolist(),
                                "generated_ids": completion, "completion": None if args.tiny else tokenizer.decode(completion, skip_special_tokens=True),
                                "wall_seconds": elapsed, "seed": args.seed, "overrides": {"max_new_tokens": args.max_new_tokens},
                                "stats": dict(model.last_generation_stats), "memory": memory()}
        save("generation_passed")
        if tokenizer is not None and not args.generation_only:
            if use_gpu:
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            probe = tokenizer(RECON_TEXT, return_tensors="pt").to(device)
            encoded = model.encode(**probe)
            count = int(encoded.latent_mask[0].sum().item())
            z = encoded.latents[0, :count]
            lengths = encoded.span_lengths[0, :count].long()
            original_ids = probe.input_ids[0]
            if int(lengths.sum().item()) != original_ids.numel() or not bool(torch.isfinite(z).all()):
                raise ValueError("non-finite encoding or incorrect span coverage")
            teacher = model(**probe).logits[0].argmax(dim=-1)
            _, duration_logits = model.vae.latent_decode(z)
            predicted_lengths = duration_logits.argmax(-1).long() + 1

            def free_decode(layout):
                condition, _ = model.vae.expand_and_refine(z, layout)
                committed = torch.empty(0, device=device, dtype=torch.long)
                offset = 0
                for width in layout.tolist():
                    logits = model.vae.span_query_logits(committed, condition[offset:offset + width], conditional=True)
                    committed = torch.cat((committed, logits.argmax(-1)))
                    offset += width
                return committed

            known = free_decode(lengths)
            predicted = free_decode(predicted_lengths)
            sync()
            spans = []
            offset = 0
            for width in lengths.tolist():
                span_ids = original_ids[offset:offset + width].tolist()
                spans.append({"token_interval": [offset, offset + width], "ids": span_ids,
                              "text": tokenizer.decode(span_ids, skip_special_tokens=False)})
                offset += width

            def decoded(tensor):
                return {"token_ids": tensor.tolist(), "text": tokenizer.decode(tensor.tolist(), skip_special_tokens=True),
                        "exact_token_match": torch.equal(tensor, original_ids)}

            record["reconstruction"] = {"text": RECON_TEXT, "input_ids": original_ids.tolist(),
                "latent_shape_without_padding": list(z.shape), "latent_dtype": str(z.dtype), "latents": z.float().tolist(),
                "span_lengths": lengths.tolist(), "spans": spans, "latent_per_token": count / original_ids.numel(),
                "teacher_forced": decoded(teacher), "free_with_encoded_span_lengths": decoded(known),
                "free_with_predicted_span_lengths": decoded(predicted), "predicted_span_lengths": predicted_lengths.tolist(),
                "free_decode_note": "Uses upstream VAE methods and greedy conditional logits (CFG 0); generated prefixes only, no original token prefix. Encoded lengths are side information in the first variant; the second uses predicted lengths only.",
                "wall_seconds": time.perf_counter() - start, "memory": memory()}
    save("passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--direct-device-load", action="store_true",
                        help="Stream checkpoint tensors to the requested GPU without a CPU model copy.")
    parser.add_argument("--sample-device-vram", action="store_true",
                        help="Record periodic device-wide AMD SMI VRAM measurements on Linux.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args()
    if args.direct_device_load and (args.device != "cuda" or args.tiny):
        parser.error("--direct-device-load requires a full GPU checkpoint run")
    if args.sample_device_vram and (args.device != "cuda" or sys.platform != "linux"):
        parser.error("--sample-device-vram requires a Linux ROCm GPU run")
    if not args.run_id.replace("-", "").isalnum():
        parser.error("use a simple run identifier")
    out = ROOT / "experiments/cola2/results" / (args.run_id + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"use a new run ID: {out}")
    record = {"started_utc": now(), "model_revision": MODEL_REV, "code_revision": CODE_REV, "args": vars(args),
              "checkpoint_loaded": False}
    sampler = None
    if args.sample_device_vram:
        from .vram_sampler import VramSampler
        sampler = VramSampler()

    def write_result():
        record["updated_utc"] = now()
        out.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")

    def save(status):
        if sampler is not None:
            sampler.phase(status)
        record["status"] = status
        write_result()
        print(status, flush=True)

    try:
        if sampler is not None:
            sampler.start()
        run(args, record, save)
    except Exception:
        record["traceback"] = traceback.format_exc()
        save("failed")
        raise
    finally:
        if sampler is not None:
            sampler.stop()
            record["device_vram_sampling"] = sampler.report()
            write_result()


if __name__ == "__main__":
    main()

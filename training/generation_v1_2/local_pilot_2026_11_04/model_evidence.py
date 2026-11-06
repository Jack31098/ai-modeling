"""Read only GGUF metadata/tensor directories and pin small official configs."""
import hashlib
import json
import math
import re
import struct
import urllib.request

from .run import BASE, ROOT, now, write


def count_scopes(inspected):
    metadata = inspected["metadata"]
    arch = metadata["general.architecture"]
    blocks = metadata[arch + ".block_count"]
    nextn = metadata.get(arch + ".nextn_predict_layers", 0)
    main_blocks = blocks - nextn
    for tensor in inspected["tensors"]:
        block_match = re.match(r"blk\.(\d+)\.", tensor["name"])
        tensor["mtp_name_or_extra_block_candidate"] = "nextn" in tensor["name"] or (block_match is not None and int(block_match[1]) >= main_blocks)
    mtp = sum(t["elements"] for t in inspected["tensors"] if t["mtp_name_or_extra_block_candidate"])
    inspected.update(main_block_count_inferred=main_blocks, mtp_candidate_tensor_elements=mtp,
                     excluding_mtp_candidate_elements=inspected["stored_tensor_elements"] - mtp,
                     mtp_scope_method="GGUF block_count includes nextn_predict_layers; blocks at or above block_count-nextn and explicit nextn names are optional MTP candidates; checked against official num_hidden_layers")
    return inspected


def inspect_gguf(path):
    formats = {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}
    budget = 64 * 1024 * 1024
    with path.open("rb") as handle:
        def read(n):
            if n < 0 or handle.tell() + n > budget:
                raise ValueError("GGUF metadata read exceeds 64MiB budget")
            data = handle.read(n)
            if len(data) != n:
                raise ValueError("truncated GGUF")
            return data

        def scalar(kind):
            fmt = "<" + formats[kind]
            return struct.unpack(fmt, read(struct.calcsize(fmt)))[0]

        def string():
            return read(scalar(10)).decode("utf-8")

        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                subtype, count = scalar(4), scalar(10)
                if count > 2_000_000:
                    raise ValueError("oversized metadata array")
                if subtype == 8:
                    for _ in range(count):
                        n = scalar(10)
                        if handle.tell() + n > budget:
                            raise ValueError("out of budget")
                        handle.seek(n, 1)
                else:
                    n = struct.calcsize("<" + formats[subtype]) * count
                    if handle.tell() + n > budget:
                        raise ValueError("out of budget")
                    handle.seek(n, 1)
                return {"array_type": subtype, "count": count, "contents": "not loaded"}
            return scalar(kind)

        if read(4) != b"GGUF":
            raise ValueError("not GGUF")
        version, ntensors, nmeta = scalar(4), scalar(10), scalar(10)
        if version not in (2, 3) or ntensors > 100_000 or nmeta > 10_000:
            raise ValueError("unexpected header")
        metadata = {}
        for _ in range(nmeta):
            key, kind = string(), scalar(4)
            item = value(kind)
            if not key.startswith("tokenizer."):
                metadata[key] = item
        metadata_end = handle.tell()
        tensors = []
        for _ in range(ntensors):
            name, ndims = string(), scalar(4)
            if not 1 <= ndims <= 4:
                raise ValueError("unexpected tensor rank")
            shape = [scalar(10) for _ in range(ndims)]
            qtype, offset = scalar(4), scalar(10)
            tensors.append({"name": name, "shape_gguf_order": shape, "elements": math.prod(shape),
                            "ggml_type": qtype, "data_offset_declared": offset})
        directory_end = handle.tell()
        handle.seek(0)
        prefix_sha = hashlib.sha256(read(directory_end)).hexdigest()
    total = sum(t["elements"] for t in tensors)
    return count_scopes({"gguf_version": version, "tensor_count": ntensors, "metadata_end_byte": metadata_end,
            "tensor_directory_end_byte": directory_end, "metadata_and_tensor_directory_sha256": prefix_sha,
            "tensor_data_read": False, "metadata": metadata, "tensors": tensors,
            "stored_tensor_elements": total,
            "count_method": "sum(product(declared tensor dimensions)); not file size divided by bytes; not a claim that buffers, tied/shared weights or optional MTP are all independently trainable/executed"})


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent":"ai-modeling-teacher-evidence/1"}), timeout=45) as response:
        raw = response.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError("metadata response too large")
    return raw, json.loads(raw)


def summary_row(model, evidence, definition, directory):
    return {**definition, "model_id": model["id"],
            "official": evidence["official"], "conversion": model,
            "gguf_stored_tensor_elements": evidence["local_gguf"]["stored_tensor_elements"],
            "gguf_optional_mtp_candidate_elements": evidence["local_gguf"]["mtp_candidate_tensor_elements"],
            "gguf_excluding_mtp_candidate_elements": evidence["local_gguf"]["excluding_mtp_candidate_elements"],
            "hf_architectures": json.loads((directory / evidence["official"]["config_file"]).read_text(encoding="utf-8"))["architectures"],
            "gguf_architecture": evidence["local_gguf"]["metadata"]["general.architecture"]}


def main():
    lock = json.loads((BASE / "assets.lock.json").read_text(encoding="utf-8"))
    directory = BASE / "model-evidence"
    directory.mkdir(exist_ok=True)
    summary = []
    definitions = {
      "qwen36-35b-a3b-q8": {"full_name":"Qwen3.6-35B-A3B", "architecture":"40-layer hybrid Gated DeltaNet / gated attention with MoE FFNs; 256 routed experts, top-8 plus one shared",
                            "official_nominal_language_parameters":35_000_000_000, "official_nominal_activated_per_token":3_000_000_000, "activation_scope":"official rounded language-model MoE claim, not an exact tensor-touch count"},
      "qwen38-27b-q8": {"full_name":"Qwen3.8-27B", "architecture":"64-layer dense-FFN hybrid Gated DeltaNet / gated attention; no routed MoE",
                        "official_nominal_language_parameters":27_000_000_000, "official_nominal_activated_per_token":None, "activation_scope":"not applicable as a sparse-MoE statistic; dense backbone participates at every token; do not equate whole stored bundle including unused MTP with per-token activation"},
      "glm47-flash-q8": {"full_name":"GLM-4.7-Flash", "architecture":"Glm4MoeLite decoder; 47 main layers, first layer dense, 64 routed experts with top-4 plus one shared in MoE layers",
                        "official_nominal_language_parameters":30_000_000_000, "official_nominal_activated_per_token":3_000_000_000, "activation_scope":"official rounded 30B-A3B claim, not inferred from Q8 file bytes"}
    }
    for model in lock["models"]:
        target = directory / (model["id"] + ".json")
        if target.exists():
            existing = json.loads(target.read_text(encoding="utf-8"))
            existing["local_gguf"] = count_scopes(existing["local_gguf"])
            write(target, existing)
            print("KEEP raw evidence, refresh derived scopes", target.name,
                  existing["local_gguf"]["stored_tensor_elements"], existing["local_gguf"]["mtp_candidate_tensor_elements"], flush=True)
            evidence = existing
            summary.append(summary_row(model, evidence, definitions[model["id"]], directory))
            continue
        inspected = inspect_gguf(ROOT / model["local_path"])
        api_url = "https://huggingface.co/api/models/" + model["upstream"]
        _, api = get_json(api_url)
        revision = api["sha"]
        config_url = "https://huggingface.co/" + model["upstream"] + "/resolve/" + revision + "/config.json"
        raw, config = get_json(config_url)
        config_name = model["id"] + "-official-config.json"
        (directory / config_name).write_bytes(raw)
        write(target, {"observed_utc": now(), "model_id": model["id"], "conversion": model,
                       "official": {"repository": model["upstream"], "revision_observed": revision,
                                    "model_card_url": "https://huggingface.co/" + model["upstream"] + "/blob/" + revision + "/README.md",
                                    "metadata_url": api_url, "safetensors_metadata": api.get("safetensors"),
                                    "config_url": config_url, "config_file": config_name,
                                    "config_sha256": hashlib.sha256(raw).hexdigest(),
                                    "snapshot_note": "official metadata observed separately; conversion's exact upstream commit is not attested by this lookup"},
                       "local_gguf": inspected})
        print(model["id"], "stored elements", inspected["stored_tensor_elements"], "MTP candidates", inspected["mtp_candidate_tensor_elements"], "official safetensors", api.get("safetensors"), flush=True)
        summary.append(summary_row(model, json.loads(target.read_text(encoding="utf-8")), definitions[model["id"]], directory))
    if len(summary) == len(lock["models"]):
        write(BASE / "parameter-summary.json", {"count_policy":"Official rounded language-model claims, full published HF safetensors bundle totals, and exact local GGUF stored tensor-element counts are distinct scopes. Counts come from official cards/configs and tensor shapes, never quantized file size. Optional MTP is not enabled by this run's speculative.types=none.", "models":summary})


if __name__ == "__main__":
    main()

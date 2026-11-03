"""Read local tokenizer metadata; no weights loaded, no model selected implicitly."""

import argparse
import json
from pathlib import Path
import struct
import subprocess

from training.word_boundary_v0.io_utils import json_bytes, sha, write_identical_or_new

ROOT = Path(__file__).resolve().parents[2]


def gguf_metadata(path):
    budget = 64 * 1024 * 1024
    scalars = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    with Path(path).open("rb") as handle:
        def read(size):
            if size < 0 or handle.tell() + size > budget:
                raise ValueError("metadata exceeds read budget")
            raw = handle.read(size)
            if len(raw) != size:
                raise ValueError("truncated GGUF metadata")
            return raw

        def scalar(kind):
            fmt = "<" + scalars[kind]
            return struct.unpack(fmt, read(struct.calcsize(fmt)))[0]

        def string(keep=True):
            size = scalar(10)
            if keep:
                return read(size).decode("utf-8")
            if handle.tell() + size > min(budget, Path(path).stat().st_size):
                raise ValueError("out-of-budget string")
            handle.seek(size, 1)
            return None

        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                subtype, count = scalar(4), scalar(10)
                if count > 2_000_000 or subtype not in (*scalars, 8):
                    raise ValueError("unsupported or oversized metadata array")
                if subtype == 8:
                    for _ in range(count):
                        string(False)
                else:
                    size = struct.calcsize("<" + scalars[subtype]) * count
                    if handle.tell() + size > budget:
                        raise ValueError("array exceeds budget")
                    handle.seek(size, 1)
                return {"array_type": subtype, "count": count, "contents": "not loaded"}
            if kind not in scalars:
                raise ValueError("unknown GGUF value type")
            return scalar(kind)

        if read(4) != b"GGUF":
            raise ValueError("not GGUF")
        version, tensors, count = scalar(4), scalar(10), scalar(10)
        if version not in (2, 3) or count > 10000:
            raise ValueError("unsupported GGUF header")
        selected = {}
        for _ in range(count):
            key, kind = string(), scalar(4)
            item = value(kind)
            if key.startswith(("tokenizer.", "general.")) or "expert" in key or "block_count" in key:
                selected[key] = item
        end = handle.tell()
        handle.seek(0)
        header_sha = sha(read(end))
    return {"version": version, "tensor_count_declared": tensors, "metadata_entries": count,
            "metadata_end_byte": end, "metadata_sha256": header_sha, "selected": selected,
            "tensor_data_read": False, "actual_qwen_tokenization": "unverified; GGUF metadata only"}


def inspect(hnet_python=None):
    locks = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "model-locks").glob("*.json"))}
    assets = []
    for p in sorted((ROOT / "models").rglob("*")):
        if p.is_file() and ("tokenizer" in p.name.lower() or p.name == "config.json" or p.suffix == ".gguf"):
            assets.append({"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
                           "sha256": sha(p.read_bytes()) if p.stat().st_size < 1_000_000 else "not-rehashed-this-inventory"})
    qwen = locks["qwen3.6-35b-a3b-q8_0.json"]
    gguf = ROOT / qwen["local_relative_path"]
    hnet = ROOT / "models/hnet_2stage_L/source/hnet__utils__tokenizers.py"
    actual = {"status": "not-run"}
    if hnet_python:
        # Executes only the already downloaded official ByteTokenizer source, no model imports.
        script = """import importlib.util,json,sys,numpy
p=sys.argv[1]; spec=importlib.util.spec_from_file_location('official_hnet_tokenizer',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m); t=m.ByteTokenizer()
texts=['ASCII 3719.','甲没有批准😀','Cafe\\u0301 与𠀀']
rows=[]
for text in texts:
 ids=t.encode([text],add_bos=False,add_eos=False)[0]['input_ids'].tolist()
 rows.append({'text':text,'ids':ids,'native_byte_spans':[[i,i+1] for i in range(len(ids))],
 'roundtrip':t.decode(ids)==text,'matches_utf8_bytes':ids==list(text.encode('utf-8'))})
print(json.dumps({'status':'actual-official-HNet-ByteTokenizer','numpy':numpy.__version__,
 'bos_eos_added':False,'examples':rows,'target_3B_tokenizer_verified':False},ensure_ascii=False))
"""
        completed = subprocess.run([str(hnet_python), "-X", "utf8", "-c", script, str(hnet)], check=True,
                                   capture_output=True, text=True, encoding="utf-8", timeout=25)
        actual = json.loads(completed.stdout)
        actual["source_sha256"] = sha(hnet.read_bytes())
        actual["code_revision"] = locks["hnet-2stage-l.json"]["hnet_code_revision"]
        actual["offset_note"] = "Native UTF-8 bytes; individual tokens can split a code point. This is not the v0 strict code-point-aligned diagnostic adapter."
    cache = Path.home() / ".cache/huggingface/hub"
    cache_dirs = sorted(p.name for p in cache.glob("models--*")) if cache.exists() else []
    return {"client_date": "2026-11-03", "scope": "repo model-locks, models filename inventory, default Hugging Face hub directory names; no credential reads or global disk search",
            "locks": locks, "local_assets": assets, "default_hf_hub_model_directories": cache_dirs,
            "qwen_gguf": gguf_metadata(gguf) if gguf.exists() else {"status": "absent"},
            "hnet_actual_tokenizer": actual,
            "target_model": {"status": "unselected", "dense_3B_source": "not established by repository evidence",
                             "warning": "Qwen 35B-A3B is MoE total size with about 3B active; not a dense 3B base. Existing HNet is not the selected training tokenizer.",
                             "current_training_intent": "B first; F and final decoder from scratch; 0.8B target scale"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hnet-python", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = inspect(args.hnet_python)
    write_identical_or_new(args.out, json_bytes(report))
    print(json.dumps({"out": str(args.out), "hnet_status": report["hnet_actual_tokenizer"]["status"],
                      "qwen_metadata_end": report["qwen_gguf"].get("metadata_end_byte")}))

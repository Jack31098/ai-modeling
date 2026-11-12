"""Download only the official inference assets; verify every published SHA-256."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile

from training.generation_v1_2.local_pilot_2026_11_04.download import fetch, now

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "models/cola2"
REPO = "ByteDance-Seed/Cola2"
MODEL_REV = "db876b905cf719b60da33c12a75021ac111f7a67"
CODE_REV = "085817168c9d6a718812fbc0075b968fc4cd5f9f"
SMALL = {"README.md", "config.json", "configuration_cola2.py", "generation_config.json", "merges.txt",
         "model.safetensors.index.json", "modeling_cola2.py", "modeling_layers.py", "special_tokens_map.json",
         "tokenizer.json", "tokenizer_config.json", "vocab.json"}
SHARDS = {f"model-{i:05}-of-00008.safetensors" for i in range(1, 9)}


def get(url, limit=16 * 1024 * 1024):
    req = urllib.request.Request(url, headers={"User-Agent": "ai-modeling-cola2-local/1"})
    with urllib.request.urlopen(req, timeout=45) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("unexpectedly large metadata/source response")
    return data


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"existing file differs: {path}")
    else:
        path.write_bytes(data)


def acquire_model(workers):
    api_url = f"https://huggingface.co/api/models/{REPO}/revision/{MODEL_REV}?blobs=true"
    metadata = json.loads(get(api_url))
    if metadata["sha"] != MODEL_REV or metadata.get("gated") or metadata.get("private"):
        raise ValueError("unexpected revision or access state")
    url = f"https://huggingface.co/{REPO}/resolve/{MODEL_REV}/"
    raw_sums = get(url + "SHA256SUMS")
    sums = {}
    for line in raw_sums.decode("ascii").splitlines():
        match = re.fullmatch(r"([a-f0-9]{64})\s+([^\s]+)", line)
        if not match:
            raise ValueError("invalid official checksum line")
        sums[match[2]] = match[1]
    if not (SMALL | SHARDS).issubset(sums):
        raise ValueError("missing inference checksum")
    files = {x["rfilename"]: x for x in metadata["siblings"]}
    folder = f"models/cola2/hf-{MODEL_REV[:12]}"
    entries = []
    for name in sorted(SMALL) + sorted(SHARDS):
        item = files[name]
        if name in SHARDS and item["lfs"]["sha256"] != sums[name]:
            raise ValueError("Hugging Face LFS digest differs from release checksum")
        entries.append({"id": name, "url": url + name, "local_path": folder + "/" + name,
                        "size_bytes": item["size"], "sha256": sums[name]})
    put(ROOT / folder / "SHA256SUMS", raw_sums)
    lock = {"repository": REPO, "model_revision": MODEL_REV, "github_code_revision": CODE_REV,
            "metadata_url": api_url, "safetensors": metadata.get("safetensors"),
            "card_license_field": (metadata.get("cardData") or {}).get("license"),
            "gated": metadata.get("gated"), "snapshot": folder, "files": entries,
            "weights_bytes": sum(e["size_bytes"] for e in entries if e["id"] in SHARDS),
            "scope": "inference allowlist only; no training or evaluation data"}
    put(LOCAL / "assets.lock.json", (json.dumps(lock, indent=2) + "\n").encode())
    for entry in entries:
        if entry["id"] in SMALL:
            fetch(entry)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        receipts = list(pool.map(fetch, (e for e in entries if e["id"] in SHARDS)))
    receipt = {"verified_at_utc": now(), "shards": receipts, "all_small_assets_verified": True}
    (LOCAL / "verified.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("ALL OFFICIAL INFERENCE FILES VERIFIED", lock["weights_bytes"], flush=True)


def acquire_source():
    url = f"https://codeload.github.com/{REPO}/zip/{CODE_REV}"
    raw = get(url)
    target = (LOCAL / f"source-{CODE_REV[:12]}").resolve()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            prefix, relative = info.filename.split("/", 1)
            if prefix != "Cola2-" + CODE_REV:
                raise ValueError("unexpected upstream source archive root")
            dest = (target / relative).resolve()
            if not dest.is_relative_to(target) or info.file_size > 4 * 1024 * 1024:
                raise ValueError("unexpected upstream archive file")
            put(dest, archive.read(info))
    put(LOCAL / "source-receipt.json", (json.dumps({"url": url, "revision": CODE_REV,
        "archive_sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw),
        "local_path": str(target.relative_to(ROOT))}, indent=2) + "\n").encode())
    print("SOURCE READY", target, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", choices=("model", "source"))
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=3)
    args = parser.parse_args()
    if args.asset == "model":
        acquire_model(args.workers)
    else:
        acquire_source()

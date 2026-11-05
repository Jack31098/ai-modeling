"""Resume hash-locked local assets. Existing complete assets are never downloaded again."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import zipfile

BASE = Path(__file__).parent
ROOT = BASE.parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def fetch(entry):
    path = ROOT / entry["local_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    reused = path.exists()
    if path.exists():
        if path.stat().st_size != entry["size_bytes"] or sha(path) != entry["sha256"]:
            raise ValueError("existing completed file differs; preserve and investigate: " + str(path))
        print("REUSED verified " + entry["id"], flush=True)
    else:
        partial = path.with_suffix(path.suffix + ".partial")
        start_time, last_log = time.monotonic(), 0
        for attempt in range(12):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > entry["size_bytes"]:
                raise ValueError("oversized partial asset")
            if offset == entry["size_bytes"]:
                break
            headers = {"User-Agent": "ai-modeling-local-teacher-pilot/1", "Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            url = entry["url"] + ("&" if "?" in entry["url"] else "?") + "download=true&pilot_retry=" + str(attempt)
            print(f"DOWNLOAD {entry['id']} resume={offset} attempt={attempt+1}", flush=True)
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45) as response:
                    if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                        raise RuntimeError("server did not honor exact byte-range resume")
                    with partial.open("ab") as target:
                        while True:
                            block = response.read(4 * 1024 * 1024)
                            if not block:
                                break
                            target.write(block)
                            offset += len(block)
                            if offset > entry["size_bytes"]:
                                raise ValueError("asset larger than locked size")
                            if offset - last_log >= 1024**3:
                                last_log = offset
                                print(f"PROGRESS {entry['id']} {offset/1e9:.2f}/{entry['size_bytes']/1e9:.2f} GB elapsed={time.monotonic()-start_time:.0f}s", flush=True)
                if offset == entry["size_bytes"]:
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
                print(f"RETRY {entry['id']} {type(error).__name__}: {error}", flush=True)
                time.sleep(2)
        if not partial.exists() or partial.stat().st_size != entry["size_bytes"] or sha(partial) != entry["sha256"]:
            raise ValueError("incomplete or checksum-failing asset; partial retained: " + str(partial))
        partial.rename(path)
        print("VERIFIED " + entry["id"], flush=True)
    return {"id": entry["id"], "path": entry["local_path"], "size_bytes": path.stat().st_size,
            "sha256": entry["sha256"], "reused": reused, "verified_at_utc": now()}


def extract(entry):
    dest = (ROOT / entry["extract_path"]).resolve()
    workspace = ROOT.resolve()
    if not dest.is_relative_to(workspace / "models"):
        raise ValueError("extraction outside workspace model tools")
    with zipfile.ZipFile(ROOT / entry["local_path"]) as archive:
        for member in archive.infolist():
            target = (dest / member.filename).resolve()
            if not target.is_relative_to(dest):
                raise ValueError("unsafe zip path")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            data = archive.read(member)
            if target.exists() and target.read_bytes() != data:
                raise FileExistsError("different tool already exists")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(data)
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("backend", "models", "all"), default="all")
    args = parser.parse_args()
    lock = json.loads((BASE / "assets.lock.json").read_text(encoding="utf-8"))
    entries = ([lock["backend"]] if args.only in ("backend", "all") else []) + (lock["models"] if args.only in ("models", "all") else [])
    for entry in entries:
        result = fetch(entry)
        if entry is lock["backend"]:
            result["extract_path"] = str(extract(entry))
        destination = ROOT / "models/local-teacher-pilot/verifications" / (entry["id"] + ".json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

"""Fingerprint installed packages/binaries and verify the unchanged checkpoint."""

from .bootstrap import RUNTIME

import argparse
import importlib.metadata as im
import json
from pathlib import Path
import subprocess
import sys

from .validate import MODEL, metadata, save, sha


def collect():
    distributions = []
    for dist in sorted(im.distributions(), key=lambda d: d.metadata["Name"].casefold()):
        record = dist.read_text("RECORD")
        direct = dist.read_text("direct_url.json")
        distributions.append({"name": dist.metadata["Name"], "version": dist.version,
                              "installed_record_sha256": __import__("hashlib").sha256(record.encode()).hexdigest() if record else None,
                              "direct_url": json.loads(direct) if direct else None})
    site = Path(sys.prefix) / "Lib/site-packages"
    paths = [site / "_rocm_sdk_core/bin/amd_comgr0702.dll", site / "_rocm_sdk_core/bin/amdhip64_7.dll",
             site / "torch/lib/c10_hip.dll", site / "torch/lib/torch_hip.dll", site / "torch/lib/torch_cpu.dll"]
    binaries = [{"path": p.relative_to(Path(sys.prefix)).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)} for p in paths]
    lock = json.loads(Path("model-locks/hnet-2stage-l.json").read_text(encoding="utf-8"))
    checkpoint = MODEL / "hnet_2stage_L.pt"
    digest = sha(checkpoint)
    assert digest == lock["sha256"] and checkpoint.stat().st_size == lock["size_bytes"]
    return {**metadata(), "status": "passed", "packages": distributions, "binaries": binaries,
            "checkpoint_sha256_after_training_tests": digest, "checkpoint_unchanged": True,
            "official_rocm_find_links": "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/",
            "pip_freeze_all": subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"], text=True),
            "compiler_candidates_in_sdk": [p.relative_to(site).as_posix() for p in (site / "_rocm_sdk_core").rglob("*.exe") if "clang" in p.name or "hipcc" in p.name],
            "note": "Versions/source index and installed RECORD/binary digests captured; this is not a hash-enforced original-wheel lock. No new weights saved."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = collect()
    save(args.out, result)
    print(json.dumps({"status": result["status"], "packages": len(result["packages"]), "checkpoint_unchanged": result["checkpoint_unchanged"]}))

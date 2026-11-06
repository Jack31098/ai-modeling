"""Check saved acceptance evidence and publish/verify a byte-level artifact manifest."""

import argparse
import json
from pathlib import Path

from .validate import BASE, save, sha

REPORTS = ["preflight", "operators", "full-cpu-serial", "full-hip-training", "cpu-hip-parity",
           "full-cpu-bf16-dechunk-serial", "full-hip-bf16-dechunk-training", "cpu-hip-bf16-dechunk-parity",
           "official-graph-cpu", "official-graph-hip", "environment-lock"]


def audit():
    failures = []
    for name in REPORTS:
        path = BASE / "results" / (name + "-2026-11-03.json")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["status"] != "passed":
            failures.append(name + ":status")
        for filename, digest in report["implementation_sha256"].items():
            if sha(BASE / filename) != digest:
                failures.append(name + ":code:" + filename)
        if sha("experiments/hnet/inspect_hierarchy.py") != report["reused_encoder_sha256"]:
            failures.append(name + ":encoder_source")
        if "training" in report and report["training"]:
            if report["parameter_count"] != 874315264 or len(report["backward"]["gradients"]) != 328:
                failures.append(name + ":incomplete-model")
            if report["optimizer"]["state_numel"] != report["parameter_count"] or report["optimizer"]["state_devices"] != ["cuda:0"]:
                failures.append(name + ":optimizer-scope")
    lock = json.loads((BASE / "source-lock.json").read_text(encoding="utf-8"))
    for entry in lock["files"]:
        if sha(entry["local_path"]) != entry["sha256"]:
            failures.append("upstream:" + entry["local_path"])
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", type=Path)
    group.add_argument("--verify", type=Path)
    args = parser.parse_args()
    errors = audit()
    if args.verify:
        manifest = json.loads(args.verify.read_text(encoding="utf-8"))
        errors += ["file:" + name for name, entry in manifest["files"].items() if sha(BASE / name) != entry["sha256"]]
    else:
        files = {p.relative_to(BASE).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)}
                 for p in sorted(BASE.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.resolve() != args.write.resolve()}
        manifest = {"client_date": "2026-11-03", "acceptance_scope": "complete 874M portable HIP reference, batch1 short sequence, full-parameter backward/SGD one step; numerical checks",
                    "status": "passed" if not errors else "failed", "checks": len(REPORTS), "files": files,
                    "errors": errors, "native_fused": "not-run", "inference_cache": "not-run", "rocm10.1": "not-run"}
        save(args.write, manifest)
    print(json.dumps({"checks": len(REPORTS), "files": len(manifest["files"]), "errors": errors, "status": "passed" if not errors else "failed"}))
    if errors:
        raise SystemExit(1)

"""Run tests and check an actual release, saving reviewable execution evidence."""

import argparse
from datetime import date
import io
import json
from pathlib import Path
import platform
import sys
import unittest

from .boundaries import BASE, dependency_info
from .io_utils import json_bytes, sha, write_identical_or_new


def verify(release, report_path, client_date):
    date.fromisoformat(client_date)
    stream = io.StringIO()
    suite = unittest.TestLoader().discover(str(BASE / "tests"))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    log = stream.getvalue()
    sys.stdout.write(log)
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    mismatches = [name for name, entry in manifest["files"].items() if not (release / name).exists() or sha((release / name).read_bytes()) != entry["sha256"]]
    implementation_mismatches = [name for name, expected in manifest["implementation_sha256"].items() if not (BASE / name).exists() or sha((BASE / name).read_bytes()) != expected]
    input_ok = sha((BASE / "inputs/documents.jsonl").read_bytes()) == manifest["input_sha256"]
    report = {"client_date": client_date, "timezone": "America/Los_Angeles", "command": ["python", "-X", "utf8", "-m", "training.word_boundary_v0.verify", "--release", str(release), "--report", str(report_path), "--client-date", client_date], "python": platform.python_version(), "dependencies": dependency_info(), "device": "CPU, jieba single-process; no model or GPU execution", "tests_run": result.testsRun, "test_sources_sha256": {p.name: sha(p.read_bytes()) for p in sorted((BASE / "tests").glob("*.py"))}, "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped), "release_manifest_sha256": sha((release / "manifest.json").read_bytes()), "release_file_count": len(manifest["files"]), "release_hash_mismatches": mismatches, "implementation_hash_mismatches": implementation_mismatches, "input_hash_matches": input_ok, "successful": result.wasSuccessful() and not mismatches and not implementation_mismatches and input_ok, "test_log": log, "limits": "Fixture/structural tests and small real excerpts only; not human word gold, Chinese streaming policy, actual tokenizer integration, or trained student quality"}
    write_identical_or_new(report_path, json_bytes(report))
    if not report["successful"]:
        raise SystemExit(1)
    print("Release hashes and tests: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--client-date", required=True, help="User/client local run date, YYYY-MM-DD")
    args = parser.parse_args()
    verify(args.release, args.report, args.client_date)

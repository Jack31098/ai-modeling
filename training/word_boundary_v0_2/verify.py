"""Run original and incremental tests; verify both releases and measure bounded state."""

import argparse
import io
import json
from pathlib import Path
import platform
import time
import tracemalloc
import unittest

from training.word_boundary_v0.io_utils import json_bytes, sha, write_identical_or_new
from .build import baseline_errors
from .candidates import BASE
from .fixture_walkthrough import walkthrough
from .streaming import POLICY_SHA, Stream, lexicon


def stress_measure():
    lexicon()  # Fixed dictionary memory is excluded explicitly from mutable state measurement.
    rows = []
    for name, chunk, count in (("continuous_han", "研究生命起源", 40000),
                                ("unknown_han", "𠀀" * 8, 30000),
                                ("continuous_ascii", "x" * 32, 8000),
                                ("long_url", "https://" + "a" * 120, 2000),
                                ("four_byte_non_han_url", "www." + "𝔄" * 60, 2000)):
        stream, output_cp = Stream(), 0
        tracemalloc.start()
        began = time.perf_counter()
        for _ in range(count):
            for unit in stream.feed(chunk):
                output_cp += len(unit["text"])
        for unit in stream.finish():
            output_cp += len(unit["text"])
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if output_cp != len(chunk) * count or stream.max_pending_cp > 64 or stream.max_pending_byte > 256 or stream.max_delay_steps > 64:
            raise AssertionError("stress bound or reconstruction failed")
        rows.append({"case": name, "characters": output_cp, "chunk_cp": len(chunk),
                     "max_pending_cp": stream.max_pending_cp, "max_pending_utf8_bytes": stream.max_pending_byte,
                     "max_commit_age_arrival_steps": stream.max_delay_steps,
                     "traced_current_bytes": current, "traced_peak_bytes": peak,
                     "elapsed_seconds": round(time.perf_counter() - began, 3)})
    return {"cases": rows, "scope": "tracemalloc starts after lexicon initialization; fixed input chunk and lexicon excluded; emitted units consumed, not retained. Counts are Python allocations, not RSS or total model memory. Timings/allocations are run-specific."}


def verify(release, report_path):
    suite = unittest.TestSuite()
    for module in ("training.word_boundary_v0.tests.test_pipeline", "training.word_boundary_v0_2.tests.test_increment"):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(module))
    log = io.StringIO()
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    errors = [name for name, spec in manifest["files"].items() if sha((release / name).read_bytes()) != spec["sha256"]]
    code_errors = [name for name, expected in manifest["implementation_sha256"].items() if sha((BASE / name).read_bytes()) != expected]
    config_errors = [name for name, expected in manifest["configuration_sha256"].items() if sha((BASE / name).read_bytes()) != expected]
    if POLICY_SHA != manifest["policy_sha256"] or lexicon()[1] != manifest["lexicon_sha256"]:
        config_errors.append("runtime_policy_or_lexicon")
    report = {"client_date": "2026-11-03", "timezone": "America/Los_Angeles", "python": platform.python_version(),
              "command": "python -X utf8 -m training.word_boundary_v0_2.verify --release " + str(release) + " --report " + str(report_path),
              "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "elapsed_test_seconds": round(time.perf_counter() - started, 3), "test_log": log.getvalue(),
              "test_sources_sha256": {p.relative_to(BASE).as_posix(): sha(p.read_bytes()) for p in (BASE / "tests").glob("*.py")},
              "release_manifest_sha256": sha((release / "manifest.json").read_bytes()),
              "release_hash_errors": errors, "implementation_hash_errors": code_errors,
              "configuration_hash_errors": config_errors,
              "old_baseline_errors": baseline_errors(), "stress": stress_measure(),
              "fixture_walkthrough": walkthrough(), "gold_accepted": 0, "gold_accuracy": None}
    report["successful"] = result.wasSuccessful() and not errors and not code_errors and not config_errors and not report["old_baseline_errors"]
    write_identical_or_new(report_path, json_bytes(report))
    print(log.getvalue())
    print(json.dumps({k: report[k] for k in ("tests_run", "failures", "errors", "successful")}))
    if not report["successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    verify(args.release, args.report)

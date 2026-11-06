"""Build a new immutable candidate/streaming release, without rerunning old split."""

import argparse
import json
from pathlib import Path

from training.word_boundary_v0.boundaries import segment
from training.word_boundary_v0.io_utils import json_bytes, jsonl_bytes, read_jsonl, sha, write_identical_or_new
from .candidates import BASE, OLD, automatic_suggestion, duplicate_audit, review_template, source_candidates
from .streaming import POLICY, POLICY_SHA, Stream, lexicon


def baseline_errors():
    manifest = json.loads((OLD / "sample_release/manifest.json").read_text(encoding="utf-8"))
    errors = ["release:" + name for name, entry in manifest["files"].items()
              if sha((OLD / "sample_release" / name).read_bytes()) != entry["sha256"]]
    errors += ["implementation:" + name for name, expected in manifest["implementation_sha256"].items()
               if sha((OLD / name).read_bytes()) != expected]
    if sha((OLD / "inputs/documents.jsonl").read_bytes()) != manifest["input_sha256"]:
        errors.append("old_inputs")
    return errors


def build(out):
    out = Path(out)
    if out.exists():
        raise FileExistsError("new output directory required; old release and reviews must not be overwritten")
    if baseline_errors():
        raise ValueError("v0 baseline changed")
    candidates = source_candidates()
    audit = duplicate_audit(candidates)
    save = lambda path, obj: write_identical_or_new(out / path, json_bytes(obj))
    write_identical_or_new(out / "candidates.jsonl", jsonl_bytes(candidates))
    save("candidate-audit.json", audit)
    save("holdout-registry.json", {"purpose": "exclude from train/tuning; pending human review; not accepted gold",
         "families": sorted({d["family_id"] for d in candidates}),
         "source_documents": sorted(d["source_document_id"] for d in candidates),
         "excerpt_hashes": sorted(d["excerpt_sha256"] for d in candidates),
         "note": "Future ingestion must apply family/lineage and duplicate exclusions. Registry alone does not enforce a future pipeline."})
    for doc in candidates:
        key = doc["doc_id"].replace(":", "-")
        save("reviews/" + key + ".json", review_template(doc))
        save("automatic-suggestions/" + key + ".json", automatic_suggestion(doc))
    old_docs = read_jsonl(OLD / "sample_release/documents.jsonl")
    statistics, examples = [], []
    for doc in old_docs + candidates:
        key = doc["doc_id"].replace(":", "-")
        text, stream = doc["text"], Stream()
        committed, checkpoints = [], []
        # Transport windows never select their ends from word labels.
        for a in range(0, len(text), 128):
            committed.extend(stream.feed(text[a:a + 128]))
            checkpoints.append({"kind": "window_pause", "state": stream.pause()})
        if doc["true_document_end"]:
            committed.extend(stream.finish())
        final = stream.snapshot()
        reconstructed = "".join(x["text"] for x in committed) + final["pending"]
        if reconstructed != text:
            raise ValueError("stream lost original text")
        write_identical_or_new(out / "commits" / (key + ".jsonl"), jsonl_bytes(committed))
        save("states/" + key + ".json", {"checkpoints": checkpoints, "final": final,
             "end_kind": "true_eof" if doc["true_document_end"] else "source_excerpt_pause"})
        observations = [{"available_at": i, "kind": "text_arrival", "text": char}
                        for i, char in enumerate(text, 1)]
        if doc["true_document_end"]:
            observations.append({"available_at": len(text) + 1, "kind": "document_end"})
        write_identical_or_new(out / "observations" / (key + ".jsonl"), jsonl_bytes(observations))
        end = final["start_cp"]
        reference = {u["cp"][1] for u in segment(text) if u["cp"][1] <= end}
        online = {u["cp"][1] for u in committed}
        row = {"doc_id": doc["doc_id"], "cohort": "old_release" if doc in old_docs else "new_candidates",
               "language": doc["language"], "cp": len(text), "bytes": len(text.encode("utf-8")),
               "committed_units": len(committed), "pending_cp": len(final["pending"]),
               "max_pending_cp": final["max_pending_cp"], "max_pending_byte": final["max_pending_byte"],
               "max_delay_steps": final["max_delay_steps"], "budget_cut_units": sum(x["continuation_possible"] for x in committed),
               "reference_comparison": {"scope_cp": [0, end], "reference_boundary_count": len(reference),
                  "stream_boundary_count": len(online), "symmetric_difference_count": len(reference ^ online),
                  "meaning": "reference disagreement, not gold quality; pending suffix excluded"}}
        statistics.append(row)
        if doc["doc_id"] in {"fixture:chinese-ambiguity", "fixture:mixed", "fixture:one-word"}:
            examples.append({"doc_id": doc["doc_id"], "text": text, "commits": committed, "final": final})
    save("statistics.json", {"documents": statistics, "gold_accepted": 0, "gold_accuracy": None,
         "bounds": {"resident_pending_cp_max": 64, "resident_pending_utf8_bytes_max": 256,
                    "han_transient_cp_max": 6, "han_after_drain_cp_max": 5,
                    "commit_age_arrival_steps_max": 64, "scope": "pending raw text only; fixed lexicon, input chunk and output sink excluded; wall-clock latency unbounded if arrivals stop"}})
    save("examples.json", examples)
    write_identical_or_new(out / "policy.json", (BASE / "policy.json").read_bytes())
    save("sources.lock.json", json.loads((BASE / "sources.lock.json").read_text(encoding="utf-8")))
    for source in json.loads((BASE / "sources.lock.json").read_text(encoding="utf-8"))["files"]:
        write_identical_or_new(out / source["local_path"], (BASE / source["local_path"]).read_bytes())
    for name in ("DATA-NOTICE.md", "REVIEW-GUIDE.md", "tokenizer-config.example.json"):
        write_identical_or_new(out / name, (BASE / name).read_bytes())
    for name in ("DATA-NOTICE.md", "inputs/provenance.json", "sources.lock.json"):
        write_identical_or_new(out / "notices/v0" / name, (OLD / name).read_bytes())
    for license_path in sorted((OLD / "licenses").glob("*")):
        if license_path.is_file():
            write_identical_or_new(out / "notices/v0/licenses" / license_path.name, license_path.read_bytes())
    files = {p.relative_to(out).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
             for p in sorted(out.rglob("*")) if p.is_file()}
    manifest = {"version": "word-boundary-v0.2", "files": files, "policy_sha256": POLICY_SHA,
                "lexicon_sha256": lexicon()[1], "lexicon_words": len(lexicon()[0]),
                "implementation_sha256": {p.name: sha(p.read_bytes()) for p in sorted(BASE.glob("*.py"))},
                "configuration_sha256": {name: sha((BASE / name).read_bytes()) for name in ("policy.json", "sources.lock.json", "tokenizer-config.example.json")},
                "old_manifest_sha256": sha((OLD / "sample_release/manifest.json").read_bytes()),
                "old_baseline_errors": baseline_errors(), "candidate_count": len(candidates),
                "candidate_families": 2, "old_documents_run": len(old_docs), "gold_accepted": 0,
                "training_ready": False, "old_split_recomputed": False}
    save("manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = build(args.out)
    print(json.dumps({"candidates": report["candidate_count"], "old_documents": report["old_documents_run"],
                      "files": len(report["files"]), "gold_accepted": 0}))

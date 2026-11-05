"""Recompute hard-check/timing metrics from immutable local server receipts."""
import json
from collections import Counter
import statistics

from .run import BASE, RESULTS, write
from .validate import validate


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize():
    docs = {d["doc_id"]: d for d in map(json.loads, (BASE / "inputs/all100.jsonl").read_text(encoding="utf-8").splitlines())}
    summary = {"status": "hard checks only; not semantic accuracy or human gold",
               "scope": "local Q8_0 Vulkan single-slot; excludes model load from annotation throughput",
               "models": {}}
    for model in load(BASE / "assets.lock.json")["models"]:
        directory = RESULTS / model["id"]
        result_rows, metrics = [], {}
        for path in sorted((directory / "documents").glob("*.json")):
            row = load(path)
            doc = docs[row["doc_id"]]
            attempts = [load(path.parent / p) for p in row["attempt_files"]]
            for attempt in attempts:
                if "response" not in attempt:
                    continue
                choice = attempt["response"]["choices"][0]
                checked = validate(doc, choice["message"].get("content") or "")
                if choice.get("finish_reason") == "length":
                    checked.update(valid=False, errors=checked["errors"] + ["output_truncated"])
                if checked != attempt["validation"]:
                    raise RuntimeError("Revalidation drift: " + str(path))
            row["first_pass_valid"] = attempts[0]["validation"]["valid"]
            row["wall_seconds_all_attempts"] = sum(a["wall_seconds"] for a in attempts)
            row["output_tokens_all_attempts"] = sum(a.get("response", {}).get("usage", {}).get("completion_tokens", 0) for a in attempts)
            row["truncations"] = sum("output_truncated" in a["validation"]["errors"] for a in attempts)
            row["reasoning_characters"] = sum(a.get("returned_reasoning_characters", 0) for a in attempts)
            result_rows.append(row)
        for cohort in ("pilot28", "expansion72", "all100", "independent4"):
            subset = [r for r in result_rows if (cohort == "all100" or
                       cohort == "pilot28" and not r["doc_id"].startswith("expansion:") or
                       cohort == "expansion72" and r["doc_id"].startswith("expansion:") or
                       cohort == "independent4" and r["doc_id"].startswith("candidate:"))]
            total_time = sum(r["wall_seconds_all_attempts"] for r in subset)
            error_counts = Counter(error for r in subset for error in r["validation"]["errors"])
            metrics[cohort] = {"completed": len(subset), "first_pass_hard_valid": sum(r["first_pass_valid"] for r in subset),
                              "final_hard_valid": sum(r["valid"] for r in subset), "retries": sum(r["attempt_count"] - 1 for r in subset),
                              "truncations": sum(r["truncations"] for r in subset),
                              "returned_reasoning_characters": sum(r["reasoning_characters"] for r in subset),
                              "annotation_wall_seconds": total_time,
                              "median_document_seconds_including_retries": statistics.median(r["wall_seconds_all_attempts"] for r in subset) if subset else None,
                              "hard_valid_documents_per_hour": sum(r["valid"] for r in subset) / total_time * 3600 if total_time else None,
                              "output_tokens_per_wall_second": sum(r["output_tokens_all_attempts"] for r in subset) / total_time if total_time else None,
                              "final_error_counts": dict(error_counts),
                              "failed_doc_ids": [r["doc_id"] for r in subset if not r["valid"]]}
        summary["models"][model["id"]] = metrics
    write(BASE / "hard-check-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    summarize()

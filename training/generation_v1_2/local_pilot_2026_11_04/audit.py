"""Audit receipts and create a portable content-hash manifest after the run."""
import json
import math
import re

from .run import BASE, RESULTS, PARAMS, sha, write, now
from .summarize import summarize, load


def audit():
    docs = {d["doc_id"]: d for d in map(json.loads, (BASE / "inputs/all100.jsonl").read_text(encoding="utf-8").splitlines())}
    assert len(docs) == 100
    assert all(not d["training_eligible"] and d["split"] is None and not d["stream_action_forcing_allowed"] for d in docs.values())
    assert len({d["content_sha256"] for d in docs.values()}) == 100
    lock = load(BASE / "assets.lock.json")
    protocol = load(BASE / "protocol.json")
    for filename, key in (("prompt.txt", "prompt_sha256"), ("schema.json", "schema_sha256"), ("inputs/all100.jsonl", "inputs_sha256")):
        assert sha(BASE / filename) == protocol[key]
    prompt = (BASE / "prompt.txt").read_text(encoding="utf-8")
    review = load(BASE / "codex-review.json")
    parameter_summary = {m["model_id"]: m for m in load(BASE / "parameter-summary.json")["models"]}
    assert load(BASE / "test-report.json")["exit_code"] == 0
    expected_review_ids = set(load(BASE / "review-selection.json")["doc_ids"])
    assert review["status"] == "complete"
    attempts_total, documents_total, gpu_proofs = 0, 0, {}
    for model in lock["models"]:
        evidence = load(BASE / "model-evidence" / (model["id"] + ".json"))
        official = evidence["official"]
        assert sha(BASE / "model-evidence" / official["config_file"]) == official["config_sha256"]
        config = load(BASE / "model-evidence" / official["config_file"])
        gguf = evidence["local_gguf"]
        assert evidence["conversion"] == model and not gguf["tensor_data_read"]
        assert gguf["tensor_count"] == len(gguf["tensors"])
        assert all(t["elements"] == math.prod(t["shape_gguf_order"]) for t in gguf["tensors"])
        assert gguf["stored_tensor_elements"] == sum(t["elements"] for t in gguf["tensors"])
        assert gguf["main_block_count_inferred"] == config.get("text_config", config)["num_hidden_layers"]
        assert parameter_summary[model["id"]]["gguf_stored_tensor_elements"] == gguf["stored_tensor_elements"]
        assert parameter_summary[model["id"]]["gguf_excluding_mtp_candidate_elements"] == gguf["excluding_mtp_candidate_elements"]
        directory = RESULTS / model["id"]
        rows = [load(p) for p in (directory / "documents").glob("*.json")]
        assert {r["doc_id"] for r in rows} == set(docs)
        reviewed = review["models"][model["id"]]["reviewed_cases"]
        assert len(reviewed) == len(expected_review_ids) and {r["doc_id"] for r in reviewed} == expected_review_ids
        results_by_id = {r["doc_id"]: r for r in rows}
        assert all(r["hard_valid"] == results_by_id[r["doc_id"]]["valid"] for r in reviewed)
        for row in rows:
            doc = docs[row["doc_id"]]
            assert row["source_sha256"] == doc["content_sha256"]
            assert row["model_id"] == model["id"] and not row["training_eligible"]
            assert row["attempt_count"] in (1, 2) and row["selected_attempt"] == row["attempt_count"]
            documents_total += 1
            for i, relative in enumerate(row["attempt_files"]):
                attempt = load(directory / "documents" / relative)
                req = attempt["request"]
                assert all(req[k] == v for k, v in PARAMS.items())
                assert req["messages"][0] == {"role": "system", "content": prompt}
                assert json.loads(req["messages"][1]["content"]) == {"text": doc["text"]}
                assert req["response_format"]["json_schema"]["schema"] == load(BASE / "schema.json")
                assert len(req["messages"]) == 2 + 2 * i
                assert attempt["attempt"] == i + 1
                attempts_total += 1
            assert row["validation"] == attempt["validation"]
            assert row["valid"] == attempt["validation"]["valid"]
        sessions = [load(p) for p in directory.glob("server-*.json")]
        assert sessions and all("stopped_utc" in s for s in sessions)
        assert all(s["model_asset"] == model and s["protocol_sha256"] == sha(BASE / "protocol.json") for s in sessions)
        assert load(directory / "asset-verification.json")["sha256"] == model["sha256"]
        logs = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in directory.glob("server-*.log"))
        full_offloads = [(int(a), int(b)) for a, b in re.findall(r"offloaded (\d+)/(\d+) layers to GPU", logs) if a == b]
        assert full_offloads and "using device Vulkan0 (AMD Radeon(TM) 8060S Graphics)" in logs
        gpu_proofs[model["id"]] = {"device": "Vulkan0 AMD Radeon 8060S", "complete_offload_log_counts": full_offloads,
                                    "log_scope": "at least one complete load per model; first Qwen3.6 launch had lower log verbosity"}
    summarize()
    write(BASE / "audit.json", {"audited_utc": now(), "documents": documents_total, "attempts": attempts_total,
          "checks": ["100 unique fixed contents per model", "300 complete model/document records", "source hashes",
                     "identical prompt/schema/sampling settings", "at most one hard-error retry", "raw response revalidation",
                     "no training admission or stream forcing", "all owned server sessions stopped", "asset hash receipts",
                     "official config hashes and layer counts", "GGUF tensor dimension sums and MTP scopes", "focused test report"],
          "primary_qualitative_outputs": len(expected_review_ids) * len(lock["models"]),
          "gpu_log_evidence": gpu_proofs,
          "limits": "No independent human gold, no semantic proof, no student consumer/tokenizer/event compiler verification"})
    entries = []
    for path in sorted(BASE.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.name != "manifest.json":
            entries.append({"path": path.relative_to(BASE).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)})
    write(BASE / "manifest.json", {"generated_utc": now(), "coordinate_policy": "original UTF-8; no newline normalization",
                                   "files": entries})
    print(f"AUDITED {documents_total} records, {attempts_total} preserved attempts; {len(entries)} manifest files")


if __name__ == "__main__":
    audit()

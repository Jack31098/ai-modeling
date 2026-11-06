"""Source-exact independent holdout candidates; automatic suggestions are not gold."""

import json
from pathlib import Path
import re

from training.word_boundary_v0.boundaries import byte_map, segment
from training.word_boundary_v0.io_utils import read_jsonl, sha
from training.word_boundary_v0.splitting import audit_normalize, shingles

BASE = Path(__file__).parent
OLD = BASE.parent / "word_boundary_v0"


def source_candidates():
    lock = json.loads((BASE / "sources.lock.json").read_text(encoding="utf-8"))
    docs = []
    for entry in lock["files"]:
        raw = (BASE / entry["local_path"]).read_bytes()
        if len(raw) != entry["bytes"] or sha(raw) != entry["sha256"]:
            raise ValueError("locked source drift")
        if entry["path"] == "LICENSE.md":
            if b"CC-BY-SA 2.5" not in raw:
                raise ValueError("license policy changed; quarantine required")
            continue
        text = raw.decode("utf-8")
        lang = "en" if "/en-us/" in entry["path"] else "zh"
        topic = entry["path"].split("/")[-2]
        if topic == "unicode":
            start_marker = "Before Unicode," if lang == "en" else "在 Unicode 定义之前"
            end_marker = " If you've ever" if lang == "en" else "如果你曾经"
            a = text.index(start_marker)
            b = text.index(end_marker, a)
        else:
            a = text.index("The first 128" if lang == "en" else "前 128 个")
            b = text.index("\n\n", a)
        excerpt = text[a:b]
        offsets = byte_map(text)
        family = "mdn:glossary/" + topic
        history = f"https://github.com/{entry['repository']}/commits/{entry['revision']}/{entry['path']}"
        upstream_translation_commit = re.search(r"sourceCommit:\s*([0-9a-f]{40})", text)
        docs.append({"doc_id": f"candidate:mdn:{lang}:{topic}", "text": excerpt, "language": lang,
                     "kind": "human_review_gold_candidate", "review_status": "pending", "annotator": None,
                     "training_eligible": False, "split": None, "true_document_end": False,
                     "source_document_id": f"mdn:{lang}:glossary/{topic}", "family_id": family,
                     "parallel_family_id": family, "derived_from": [entry["url"]],
                     "source": entry, "source_cp": [a, b], "source_byte": [offsets[a], offsets[b]],
                     "excerpt_sha256": sha(excerpt.encode("utf-8")), "cp_to_byte": byte_map(excerpt),
                     "translation_source_commit": upstream_translation_commit.group(1) if upstream_translation_commit else None,
                     "attribution": "MDN contributors", "attribution_history_url": history,
                     "license": "CC-BY-SA-2.5", "license_url": "https://creativecommons.org/licenses/by-sa/2.5/",
                     "license_file": "sources/" + ("en-us" if lang == "en" else "zh-cn") + "/LICENSE.md",
                     "changes": "contiguous prose excerpt only; no normalization, rewriting or translation"})
    return sorted(docs, key=lambda d: d["doc_id"])


def review_template(doc):
    return {"doc_id": doc["doc_id"], "excerpt_sha256": doc["excerpt_sha256"], "text": doc["text"],
            "coordinate_unit": "unicode_code_point", "cp_to_byte": doc["cp_to_byte"],
            "review_status": "pending", "annotator": None, "units": [], "ambiguities": [],
            "notes": "Human fills all units; automatic suggestions are stored separately."}


def validate_review(doc, review):
    if review["doc_id"] != doc["doc_id"] or review["text"] != doc["text"] or review["excerpt_sha256"] != doc["excerpt_sha256"]:
        raise ValueError("review is not for this exact candidate")
    if review["review_status"] != "accepted" or not isinstance(review["annotator"], str) or not review["annotator"].strip():
        raise ValueError("human review pending: no gold evaluation allowed")
    offsets, cursor = byte_map(doc["text"]), 0
    for unit in review["units"]:
        a, b = unit["cp"]
        if not (a == cursor and a < b <= len(doc["text"])) or unit["text"] != doc["text"][a:b] or unit["byte"] != [offsets[a], offsets[b]]:
            raise ValueError("invalid review partition or byte offsets")
        cursor = b
    if cursor != len(doc["text"]):
        raise ValueError("incomplete human review")
    if any(not x.get("resolution") for x in review["ambiguities"]):
        raise ValueError("unresolved ambiguity")
    return True


def duplicate_audit(candidates):
    inputs = read_jsonl(OLD / "inputs/documents.jsonl")
    kept = read_jsonl(OLD / "sample_release/documents.jsonl")
    train_ids = {d["doc_id"] for d in kept if d["split"] == "train"}
    rows = []
    for doc in candidates:
        matches, maximum = [], {"score": 0, "old_doc_id": None}
        norm = audit_normalize(doc["text"])
        a = shingles(norm)
        for previous in inputs:
            other = audit_normalize(previous["text"])
            b = shingles(other)
            score = len(a & b) / len(a | b) if a | b else 0
            if score > maximum["score"]:
                maximum = {"score": score, "old_doc_id": previous["doc_id"]}
            reasons = []
            if doc["text"] == previous["text"]:
                reasons.append("exact")
            if norm == other:
                reasons.append("audit_normalized")
            if re.sub(r"[0-9]+", "<NUM>", norm) == re.sub(r"[0-9]+", "<NUM>", other):
                reasons.append("number_template")
            if min(len(norm), len(other)) >= 30 and score >= .85:
                reasons.append("char5gram_jaccard_ge_0.85")
            for key in ("source_document_id", "family_id", "parallel_family_id"):
                if doc.get(key) and doc[key] == previous.get(key):
                    reasons.append(key)
            if reasons:
                matches.append({"old_doc_id": previous["doc_id"], "in_train": previous["doc_id"] in train_ids,
                                "reasons": reasons, "score": score})
        rows.append({"doc_id": doc["doc_id"], "matches": matches, "nearest_old_input": maximum,
                     "disposition": "quarantine_overlap" if matches else "isolated_pending_human_review"})
    family_links = [{"a": a["doc_id"], "b": b["doc_id"], "reason": "canonical_topic_parallel_family",
                     "family": a["family_id"], "exact_sentence_alignment_claimed": False}
                    for i, a in enumerate(candidates) for b in candidates[i + 1:] if a["family_id"] == b["family_id"]]
    return {"old_input_count": len(inputs), "old_train_count": len(train_ids), "results": rows,
            "candidate_family_links": family_links,
            "old_inputs_sha256": sha((OLD / "inputs/documents.jsonl").read_bytes()),
            "old_split_audit_sha256": sha((OLD / "sample_release/split-audit.json").read_bytes()),
            "limitations": "Exact, metadata lineage, normalized, numeric-template and char5gram audit only; unknown semantic paraphrases are not ruled out. Explicit bilingual families remain held out together."}


def automatic_suggestion(doc):
    return {"doc_id": doc["doc_id"], "status": "automatic_unreviewed_not_gold", "main_input_allowed": False,
            "policy": "v0-offline-rules-plus-jieba-DAG-HMM-false", "units": segment(doc["text"])}

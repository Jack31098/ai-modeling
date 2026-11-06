"""Build a small release with raw text, spans, isolated tapes, alignment audits and hashes."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import platform
import sys

from .alignment import align_tokens, codepoint_fixture
from .boundaries import BASE, RULES, byte_map, dependency_info, segment
from .events import compile_tapes, slice_document, training_pairs
from .io_utils import json_bytes, jsonl_bytes, read_jsonl, sha
from .splitting import assign_documents


def histogram(values):
    return dict(sorted(Counter(values).items()))


def build(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("use a new output directory; existing user data is never replaced")
    docs_path = BASE / "inputs/documents.jsonl"
    provenance = json.loads((BASE / "inputs/provenance.json").read_text(encoding="utf-8"))
    if sha(docs_path.read_bytes()) != provenance["documents_sha256"]:
        raise ValueError("input checksum differs from provenance")
    dependencies = dependency_info()
    docs, audit = assign_documents(read_jsonl(docs_path))
    files, samples, slices, stats_rows, consumption = {}, [], [], [], []
    groups = defaultdict(set)
    for doc in docs:
        groups[doc["isolation_group"]].add(doc["split"])
        text = doc["text"]
        units = segment(text)
        observations, labels = compile_tapes(doc, units)
        ends = [u["cp"][1] for u in units]
        # These are diagnostic adapters, not an actual model tokenizer.
        char_alignment = align_tokens(text, ends, codepoint_fixture(text), offset_unit="unicode_code_point", tokenizer_id="fixture:one-codepoint-v1")
        fused_alignment = align_tokens(text, ends, [{"id": "whole-input", "span": [0, len(text)], "surface": text}], offset_unit="unicode_code_point", tokenizer_id="fixture:whole-input-v1")
        if not char_alignment["usable_without_conflicts"]:
            raise ValueError("codepoint identity adapter should preserve every boundary")
        for mode in ("READ_FULL", "READ_STREAM"):
            frames, loss_events = 0, 0
            for pair in training_pairs(observations, labels, mode):
                frames += 1
                loss_events += bool(pair["loss_targets"])
                assert set(pair["model_input"]) == {"text", "input_complete", "document_ended"}
                assert "target" not in pair["model_input"]
                if mode == "READ_STREAM":
                    assert all(label["supervision_type"] == "causal_action_delayed" for label in pair["loss_targets"])
            consumption.append({"doc_id": doc["doc_id"], "mode": mode, "observation_driven_model_frames": frames, "nonempty_loss_events": loss_events, "consumer": "loss-plumbing reference, no neural network or teacher action injection"})
        key = sha(doc["doc_id"].encode("utf-8"))[:16]
        samples.append({**doc, "coordinate_unit": "unicode_code_point", "cp_to_byte": byte_map(text), "normalization": "none", "units": units, "offline_reference_label_id": "full_reference", "observation_file": f"observations/{key}.jsonl", "supervision_file": f"supervision/{key}.jsonl", "alignment_file": f"alignment/{key}.json", "tokenizer": "UNSELECTED; diagnostic fixtures only"})
        files[f"observations/{key}.jsonl"] = jsonl_bytes(observations)
        files[f"supervision/{key}.jsonl"] = jsonl_bytes(labels)
        files[f"alignment/{key}.json"] = json_bytes({"doc_id": doc["doc_id"], "identity_codepoint_fixture": char_alignment, "fused_token_conflict_fixture": fused_alignment})
        slices.extend(slice_document(doc, units, labels, RULES["slicing"]["code_points"]))
        stats_rows.append({"doc": doc, "units": units, "stream": [label for label in labels if label["mode"] == "READ_STREAM"], "alignment_conflicts": sum(not row["loss_mask"] for row in fused_alignment["boundaries"])})
    assert all(len(values) == 1 for values in groups.values())
    assert all(s["split"] == next(d["split"] for d in docs if d["doc_id"] == s["doc_id"]) for s in slices)
    statistics = {"input_documents": audit["input_documents"], "kept_documents": len(docs), "slices": len(slices), "split_documents": {}, "languages": {}, "rejections": audit["rejection_counts"], "real_corpus_quality": "not measured; two literary source documents, no modern Chinese gold or model tokenizer"}
    statistics["rejections_by_split"] = {name: dict(sorted(Counter(row["reason"] for row in audit["rejected_documents"] if (row["split"] or "unassigned_quarantine") == name).items())) for name in ("train", "dev", "test", "unassigned_quarantine")}
    statistics["corpus_vs_fixtures"] = {kind: {language: {"documents": sum(row["doc"]["kind"] == kind and row["doc"]["language"] == language for row in stats_rows), "code_points": sum(len(row["doc"]["text"]) for row in stats_rows if row["doc"]["kind"] == kind and row["doc"]["language"] == language), "units": sum(len(row["units"]) for row in stats_rows if row["doc"]["kind"] == kind and row["doc"]["language"] == language)} for language in ("en", "zh", "mixed")} for kind in ("real_corpus_excerpt", "synthetic_fixture")}
    for split in ("train", "dev", "test"):
        selected = [row for row in stats_rows if row["doc"]["split"] == split]
        statistics["split_documents"][split] = {"documents": len(selected), "isolation_groups": len({row["doc"]["isolation_group"] for row in selected}), "by_kind": dict(sorted(Counter(row["doc"]["kind"] for row in selected).items())), "by_language": dict(sorted(Counter(row["doc"]["language"] for row in selected).items()))}
    for language in ("en", "zh", "mixed"):
        selected = [row for row in stats_rows if row["doc"]["language"] == language]
        units = [u for row in selected for u in row["units"]]
        lexical = [u for u in units if u["kind"] in {"word", "han_word", "number", "url"}]
        cps = sum(len(row["doc"]["text"]) for row in selected)
        statistics["languages"][language] = {"documents": len(selected), "code_points": cps, "utf8_bytes": sum(len(row["doc"]["text"].encode("utf-8")) for row in selected), "unit_kinds": dict(sorted(Counter(u["kind"] for u in units).items())), "unit_cp_length_histogram": histogram(u["cp"][1] - u["cp"][0] for u in units), "unit_byte_length_histogram": histogram(u["byte"][1] - u["byte"][0] for u in units), "lexical_cp_length_histogram": histogram(u["cp"][1] - u["cp"][0] for u in lexical), "lexical_byte_length_histogram": histogram(u["byte"][1] - u["byte"][0] for u in lexical), "unit_boundary_rate_per_codepoint": round(len(units) / cps, 6) if cps else None, "lexical_end_rate_per_codepoint": round(len(lexical) / cps, 6) if cps else None, "sentence_terminal_candidates": sum(u["sentence_terminal_candidate"] for u in units), "delayed_stream_decisions": sum(len(row["stream"]) for row in selected), "fused_fixture_unaligned_boundaries": sum(row["alignment_conflicts"] for row in selected)}
    files["documents.jsonl"] = jsonl_bytes(samples)
    files["slices.jsonl"] = jsonl_bytes(slices)
    files["split-audit.json"] = json_bytes(audit)
    files["statistics.json"] = json_bytes(statistics)
    files["consumer-check.json"] = json_bytes({"scope": "data/permission plumbing only; B/F/decoder not trained or implemented", "observations_only_model_input": True, "group_and_slice_isolation": True, "decisions": consumption})
    files["provenance.json"] = json_bytes(provenance)
    files["rules.json"] = (BASE / "rules.json").read_bytes()
    for path in sorted((BASE / "licenses").glob("*")):
        files["licenses/" + path.name] = path.read_bytes()
    files["DATA-NOTICE.md"] = (BASE / "DATA-NOTICE.md").read_bytes()
    examples = ["Examples: lossless fixed-rule references, not human word-boundary gold.", "Offsets are Unicode code points and UTF-8 bytes; text displayed with JSON escaping."]
    for sample in samples:
        if sample["kind"] == "real_corpus_excerpt" or sample["doc_id"] in {"fixture:chinese-role", "fixture:english-punctuation", "fixture:mixed", "fixture:unicode"}:
            examples.extend(["", f"{sample['doc_id']} [{sample['kind']}, {sample['split']}]", json.dumps(sample["text"], ensure_ascii=False)])
            for unit in sample["units"][:32]:
                examples.append(f"cp={unit['cp']} byte={unit['byte']} {unit['kind']}: " + json.dumps(unit["text"], ensure_ascii=False))
            if len(sample["units"]) > 32:
                examples.append("... full units in documents.jsonl")
    files["examples.txt"] = ("\n".join(examples) + "\n").encode("utf-8")
    implementation_files = sorted(BASE.glob("*.py")) + [BASE / "sources.lock.json", BASE / "rules.json", BASE / "requirements.lock"]
    manifest = {"schema": "word-boundary-release-v0.1", "input_sha256": sha(docs_path.read_bytes()), "input_provenance_sha256": sha((BASE / "inputs/provenance.json").read_bytes()), "dependencies": dependencies, "runtime": {"python": platform.python_version(), "implementation": platform.python_implementation(), "device": "CPU; no ML model", "platform": sys.platform}, "implementation_sha256": {p.name: sha(p.read_bytes()) for p in implementation_files}, "seed": RULES["split"]["seed"], "files": {name: {"sha256": sha(data), "bytes": len(data)} for name, data in sorted(files.items())}, "training_readiness": {"offline_read_full_reference": True, "english_delayed_stream_reference": True, "chinese_stream_actions": False, "real_model_tokenizer": False, "neural_student_consumer": False, "human_gold_quality_audit": False}, "scope": "first-layer fixed word units; partial D1 preparation, not a complete v1.2 D1/D2 release"}
    files["manifest.json"] = json_bytes(manifest)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in sorted(files.items()):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return statistics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.out)
    print(json.dumps({key: result[key] for key in ("input_documents", "kept_documents", "slices", "split_documents", "rejections")}, ensure_ascii=False, indent=2))

"""Concrete v0 fixture observations for the user's explanation, not semantic scores."""

from training.word_boundary_v0.events import compile_tapes, consume_labels, model_input
from training.word_boundary_v0.io_utils import read_jsonl
from .candidates import OLD


def walkthrough():
    docs = {d["doc_id"]: d for d in read_jsonl(OLD / "sample_release/documents.jsonl")}
    doc = docs["fixture:one-word"]
    observations, labels = compile_tapes(doc, doc["units"])
    eof = [{"cutoff": t, "model_input": model_input(observations, t),
            "available_stream_label_count": len(consume_labels(labels, t, "READ_STREAM"))}
           for t in (len(doc["text"]), len(doc["text"]) + 1)]
    frames = []
    prefix = "Store the identifier 3719."
    for key in ("fixture:same-prefix-a", "fixture:same-prefix-b"):
        d = docs[key]
        obs, _ = compile_tapes(d, d["units"])
        frames.append({"doc_id": key, "model_input": model_input(obs, len(prefix)),
                       "split": d["split"], "group": d["isolation_group"]})
    d = docs["fixture:chinese-role"]
    obs, labels = compile_tapes(d, d["units"])
    return {"synthetic_input_fixture_count": 27, "original_unittest_count": 30,
            "one_word_eof_counterfactual": eof, "same_prefix": frames,
            "same_prefix_inputs_equal": frames[0]["model_input"] == frames[1]["model_input"],
            "chinese_role": {"units": d["units"], "cutoff_3_input": model_input(obs, 3),
                             "cutoff_3_stream_labels": consume_labels(labels, 3, "READ_STREAM")},
            "template_inventory": [{k: docs[f"fixture:inventory-{n}"][k] for k in ("doc_id", "text", "split", "isolation_group")} for n in (10, 17, 73)],
            "limits": "Data contracts and timing only; no trained B, semantic role, digit recall or gold quality measured."}

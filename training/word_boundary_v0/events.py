"""Observation-only consumption. Supervision is a separate, explicitly timed loss input."""

from .boundaries import online_english_chunks


def compile_tapes(doc, units):
    text, n = doc["text"], len(doc["text"])
    observations = [{"event_id": f"o{i:06d}", "available_at": i, "kind": "text_arrival", "text": c} for i, c in enumerate(text, 1)]
    observations.append({"event_id": f"o{n + 1:06d}", "available_at": n + 1, "kind": "document_end" if doc.get("true_document_end", True) else "source_excerpt_end"})
    ends = {u["cp"][1] for u in units}
    source = {"rule_version": "fixed-orthographic-v0.1", "source_id": doc["source_id"], "content_sha256": doc["content_sha256"]}
    common = {"target_module": "B_plus_boundary", "scale": 1, "main_input_allowed": False, "label_source": source, "latent_target": None, "continuous_state_substitution_allowed": False}
    labels = [{**common, "label_id": "full_reference", "mode": "READ_FULL", "supervision_type": "offline_reference", "available_at": n + 1, "evidence_available_at": n + 1, "decision_at": n + 1, "observation_cutoff": n + 1, "apply_loss_at": n + 1, "allow_stream_action_forcing": False, "target": [{"cp": p, "is_boundary": p in ends} for p in range(1, n + 1)]}]
    for index, decision in enumerate(online_english_chunks(text, doc.get("true_document_end", True))):
        time = decision["available_at"]
        labels.append({**common, "label_id": f"stream{index:06d}", "mode": "READ_STREAM", "supervision_type": "causal_action_delayed", "available_at": time, "evidence_available_at": time, "decision_at": time, "observation_cutoff": time, "apply_loss_at": time, "allow_stream_action_forcing": True, "buffer_cp": [decision["start"], decision["end"]], "target": [{"cp": p, "is_boundary": p in decision["boundaries"]} for p in range(decision["start"] + 1, decision["end"] + 1)]})
    return observations, sorted(labels, key=lambda label: (label["available_at"], label["label_id"]))


def model_input(observations, cutoff):
    """Whitelist ONLY arrived text and received end signals. Never inspect label files."""
    visible = [o for o in observations if o["available_at"] <= cutoff]
    return {"text": "".join(o["text"] for o in visible if o["kind"] == "text_arrival"), "input_complete": any(o["kind"] in {"document_end", "source_excerpt_end"} for o in visible), "document_ended": any(o["kind"] == "document_end" for o in visible)}


def consume_labels(labels, cutoff, mode):
    result = []
    for label in labels:
        if label["mode"] != mode or label["available_at"] > cutoff:
            continue
        if any(label[field] > cutoff for field in ("evidence_available_at", "decision_at", "observation_cutoff", "apply_loss_at")):
            raise ValueError("label timing violation")
        result.append(label)
    return result


def training_pairs(observations, labels, mode):
    """Reference consumer for loss plumbing only; does not execute a neural B or inject actions."""
    if mode not in {"READ_FULL", "READ_STREAM"}:
        raise ValueError("unsupported consumer mode")
    # The observation clock, not the presence/timing of labels, schedules model inputs.
    for event in observations:
        time = event["available_at"]
        frame = model_input(observations, time)
        if mode == "READ_FULL" and not frame["input_complete"]:
            continue
        targets = [label for label in consume_labels(labels, time, mode) if label["available_at"] == time]
        yield {"decision_at": time, "model_input": frame, "loss_targets": targets}


def slice_document(doc, units, labels, width):
    if width <= 0:
        raise ValueError("slice width must be positive")
    from .boundaries import byte_map
    text = doc["text"]
    offsets = byte_map(text)
    slices = []
    for start in range(0, len(text), width):
        end = min(start + width, len(text))
        cutoff = end + 1 if end == len(text) else end
        slices.append({"doc_id": doc["doc_id"], "split": doc["split"], "isolation_group": doc["isolation_group"], "cp": [start, end], "byte": [offsets[start], offsets[end]], "text": text[start:end], "requires_prefix_replay": True, "prefix_replay_cp": [0, start], "observation_cutoff": cutoff, "is_artificial_window_end": end < len(text) or not doc.get("true_document_end", True), "inject_eos": False, "stream_label_ids": [label["label_id"] for label in labels if label["mode"] == "READ_STREAM" and label["available_at"] <= cutoff and label["buffer_cp"][0] >= start and label["buffer_cp"][1] <= end], "crossing_units_masked": [u["cp"] for u in units if u["cp"][0] < end and u["cp"][1] > start and not (start <= u["cp"][0] and u["cp"][1] <= end)], "offline_reference_usage": "READ_FULL only, after original input_complete, never as streaming forced actions"})
    return slices

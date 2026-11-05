"""Hard geometry checks; lexical/semantic quality remains a separate review."""

import json
import unicodedata

from training.word_boundary_v0.boundaries import byte_map


def validate(doc, content):
    errors = []
    try:
        obj = json.loads(content)
    except (ValueError, TypeError) as exc:
        return {"valid": False, "errors": ["invalid_json: " + str(exc)]}
    if not isinstance(obj, dict) or set(obj) != {"phrases"}:
        return {"valid": False, "errors": ["expected_only_phrases_key"]}
    phrases = obj["phrases"]
    if not isinstance(phrases, list) or not 1 <= len(phrases) <= 32:
        return {"valid": False, "errors": ["phrase_count_or_type"]}
    for i, words in enumerate(phrases):
        if not isinstance(words, list) or not 1 <= len(words) <= 12:
            errors.append(f"phrase_{i}_word_count_or_type")
            continue
        if any(not isinstance(w, str) or not 1 <= len(w) <= 64 for w in words):
            errors.append(f"phrase_{i}_word_type_or_length")
            continue
        if len("".join(words)) > 64:
            errors.append(f"phrase_{i}_exceeds_64cp")
    if errors:
        return {"valid": False, "errors": errors}
    flat = [w for words in phrases for w in words]
    reconstructed = "".join(flat)
    text = doc["text"]
    if reconstructed != text:
        diff = next((i for i, (a, b) in enumerate(zip(text, reconstructed)) if a != b), min(len(text), len(reconstructed)))
        return {"valid": False, "errors": ["raw_text_mismatch"], "first_difference_cp": diff,
                "expected_length_cp": len(text), "actual_length_cp": len(reconstructed),
                "actual_text": reconstructed}
    cp_bytes, pos = byte_map(text), 0
    words_out, phrases_out, boundaries = [], [], {0}
    for pi, words in enumerate(phrases):
        start, child_ids = pos, []
        for word in words:
            end = pos + len(word)
            wid = "w" + str(len(words_out))
            words_out.append({"id": wid, "text": word, "cp": [pos, end], "utf8": [cp_bytes[pos], cp_bytes[end]]})
            child_ids.append(wid)
            if any(c.isspace() for c in word) and not word.isspace():
                errors.append(f"mixed_whitespace_word:{wid}")
            if unicodedata.category(word[0]).startswith("M") or word[0] == "\u200d":
                errors.append(f"detached_combining_or_joiner:{wid}")
            pos = end
            boundaries.add(pos)
        phrases_out.append({"id": "p" + str(pi), "cp": [start, pos], "utf8": [cp_bytes[start], cp_bytes[pos]],
                            "children": child_ids, "passthrough": len(child_ids) == 1})
    for protected in doc.get("protected_strings", []):
        cursor = 0
        while (offset := text.find(protected, cursor)) >= 0:
            if any(offset < b < offset + len(protected) for b in boundaries):
                errors.append("split_protected_string:" + protected)
            cursor = offset + len(protected)
    result = {"valid": not errors, "errors": errors, "word_count": len(flat), "phrase_count": len(phrases),
              "passthrough_count": sum(p["passthrough"] for p in phrases_out),
              "word_to_phrase_node_ratio": len(flat) / len(phrases),
              "semantic_quality": "unreviewed; exact reconstruction does not certify lexical or semantic boundaries"}
    if errors:
        return result
    # This is a candidate span projection, not a student tokenizer or neural execution trace.
    result["candidate_structure"] = {
        "coordinates": "Unicode code points and original UTF-8 bytes; half-open",
        "words": words_out, "phrase_forest": phrases_out,
        "scale_projection": {"0": [w["id"] for w in words_out], "1": [p["id"] for p in phrases_out]},
        "student_tokenizer_alignment": "not implemented",
        "execution_events": "not compiled; complete v1.2 D1 consumer still required",
        "supervision_permissions": {
            "mode": "READ_FULL", "evidence_available_at": "entire supplied input received",
            "input_complete_is_true_document_eof": doc["true_document_end"],
            "student_observation_fields": ["text", "input_complete"],
            "structure_storage": "supervision only; never append to student text",
            "allow_stream_action_forcing": False,
            "allow_continuous_latent_substitution": False,
            "allow_training": False,
        },
    }
    return result

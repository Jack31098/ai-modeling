"""Strict adapter for ORIGINAL text offsets; no snapping or implicit normalization."""

from .boundaries import byte_map


def align_tokens(text, boundaries, tokens, *, offset_unit, tokenizer_id, original_text=None):
    """Tokens: {id, span:[start,end]|None, special:bool, surface?:str}.

    Supports Unicode code points, UTF-8 bytes and UTF-16 code units. Offsets must
    reference the exact original text. 'surface' is optional decoded source text,
    not a vocab spelling like Ġword. No model/tokenizer dependency is imported.
    """
    if offset_unit not in {"unicode_code_point", "utf8_byte", "utf16_code_unit"}:
        raise ValueError("explicit supported offset_unit required")
    if len(boundaries) != len(set(boundaries)) or any(type(p) is not int or not 0 < p <= len(text) for p in boundaries):
        raise ValueError("invalid reference boundaries")
    errors, notes, spans = [], [], []
    if original_text is not None and original_text != text:
        errors.append({"reason": "normalization_mismatch"})
    utf8 = byte_map(text)
    coords = list(range(len(text) + 1)) if offset_unit == "unicode_code_point" else utf8
    if offset_unit == "utf16_code_unit":
        coords = [0]
        for char in text:
            coords.append(coords[-1] + len(char.encode("utf-16-le")) // 2)
    inverse = {value: index for index, value in enumerate(coords)}
    for index, token in enumerate(tokens):
        span = token.get("span")
        if token.get("special", False):
            if span is not None and span != [0, 0]:
                errors.append({"reason": "special_token_has_text_span", "token": index})
            notes.append({"reason": "special_token_excluded", "token": index})
            continue
        if not isinstance(span, list) or len(span) != 2 or any(type(v) is not int for v in span):
            errors.append({"reason": "missing_or_invalid_offset", "token": index})
            continue
        a, b = span
        if a not in inverse or b not in inverse:
            errors.append({"reason": "unicode_or_out_of_range_offset", "token": index, "span": span})
            continue
        a, b = inverse[a], inverse[b]
        if a >= b:
            errors.append({"reason": "empty_or_reversed_offset", "token": index})
            continue
        if "surface" in token and token["surface"] != text[a:b]:
            errors.append({"reason": "surface_mismatch", "token": index})
        spans.append((index, a, b))
    cursor = 0
    for index, a, b in spans:
        if a != cursor:
            errors.append({"reason": "token_overlap_or_order" if a < cursor else "token_coverage_gap", "token": index})
        cursor = max(cursor, b)
    if cursor != len(text):
        errors.append({"reason": "token_coverage_gap", "at": cursor})
    results = []
    for boundary in sorted(boundaries):
        interiors = [index for index, a, b in spans if a < boundary < b]
        edges = [index for index, _, b in spans if b == boundary]
        if interiors:
            reason = "boundary_inside_token"
        elif len(edges) != 1:
            reason = "ambiguous_or_missing_token_edge"
        else:
            reason = "aligned" if not errors else "invalid_adapter"
        results.append({"cp": boundary, "byte": utf8[boundary], "status": reason, "token_indices": interiors or edges, "loss_mask": reason == "aligned"})
    for index, a, b in spans:
        positions = [p for p in boundaries if a < p < b]
        if len(positions) > 1:
            errors.append({"reason": "multiple_boundaries_inside_token", "token": index, "boundaries": sorted(positions)})
    return {"tokenizer_id": tokenizer_id, "offset_unit": offset_unit, "surface_check": "verified" if all("surface" in t for t in tokens if not t.get("special")) else "adapter_geometry_only", "usable_without_conflicts": not errors and all(x["loss_mask"] for x in results), "errors": errors, "notes": notes, "boundaries": results}


def codepoint_fixture(text):
    return [{"id": f"cp:{ord(c)}", "span": [i, i + 1], "surface": c} for i, c in enumerate(text)]

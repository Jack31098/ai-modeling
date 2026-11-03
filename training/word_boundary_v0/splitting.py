"""Small-corpus conservative grouping before any slicing (quadratic audit)."""

from collections import Counter
import hashlib
import re
import unicodedata

from .boundaries import RULES


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def audit_normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def shingles(text):
    return {text[i:i + 5] for i in range(max(0, len(text) - 4))}


def assign_documents(documents):
    docs = sorted(documents, key=lambda d: d["doc_id"])
    if len(docs) > RULES["split"]["max_documents_for_pairwise_audit"]:
        raise ValueError("prototype pairwise audit limit exceeded; use a separately versioned scalable audit")
    ids = [d["doc_id"] for d in docs]
    if len(ids) != len(set(ids)):
        raise ValueError("doc_id collision")
    parent = {i: i for i in ids}
    reasons = []

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b, reason, score=None):
        x, y = find(a), find(b)
        parent[max(x, y)] = min(x, y)
        reasons.append({"a": a, "b": b, "reason": reason, **({"score": score} if score is not None else {})})

    for d in docs:
        for ancestor in d.get("derived_from", []):
            if ancestor not in parent:
                raise ValueError("unresolved derivative lineage: " + ancestor)
            union(d["doc_id"], ancestor, "derived_from")
    normalized = {d["doc_id"]: audit_normalize(d["text"]) for d in docs}
    for pos, a in enumerate(docs):
        for b in docs[pos + 1:]:
            ai, bi = a["doc_id"], b["doc_id"]
            for key in ("source_document_id", "family_id", "template_family_id", "parallel_family_id"):
                if a.get(key) and a.get(key) == b.get(key):
                    union(ai, bi, key)
            if a["text"] == b["text"]:
                union(ai, bi, "exact_duplicate")
            na, nb = normalized[ai], normalized[bi]
            if len(na) >= 30 and len(nb) >= 30:
                if re.sub(r"[0-9]+", "<NUM>", na) == re.sub(r"[0-9]+", "<NUM>", nb):
                    union(ai, bi, "number_template")
                sa, sb = shingles(na), shingles(nb)
                score = len(sa & sb) / len(sa | sb)
                if score >= 0.85:
                    union(ai, bi, "near_duplicate", round(score, 6))
    kept, rejected, exact_seen = [], [], {}
    for d in sorted(docs, key=lambda d: (len(d.get("derived_from", [])), d["doc_id"])):
        reason = None
        if not d["text"].strip():
            reason = "empty_or_whitespace_only"
        elif not d.get("redistribution_allowed", False):
            reason = "unverified_redistribution"
        elif d["text"] in exact_seen:
            reason = "exact_duplicate"
        if reason:
            rejected.append({"doc_id": d["doc_id"], "reason": reason, "disposition": "drop" if reason == "exact_duplicate" else "quarantine", "duplicate_of": exact_seen.get(d["text"]), "group": find(d["doc_id"])})
        else:
            exact_seen[d["text"]] = d["doc_id"]
            kept.append(d)
    groups = sorted({find(d["doc_id"]) for d in kept}, key=lambda g: (digest(RULES["split"]["seed"] + "|" + g), g))
    n = len(groups)
    # Group-level 80/10/10 with minimum 1 dev/test when at least 3 groups.
    ndev = max(1, n // 10) if n >= 3 else 0
    ntest = max(1, n // 10) if n >= 2 else 0
    ntrain = n - ndev - ntest
    split_by_group = {g: "train" if rank < ntrain else "dev" if rank < ntrain + ndev else "test" for rank, g in enumerate(groups)}
    for row in rejected:
        row["split"] = split_by_group.get(row["group"])
    output = []
    for d in kept:
        output.append({**d, "content_sha256": digest(d["text"]), "isolation_group": find(d["doc_id"]), "split": split_by_group[find(d["doc_id"])]})
    output.sort(key=lambda d: d["doc_id"])
    return output, {"version": "grouped-split-v0.1", "input_documents": len(docs), "kept_documents": len(output), "rejected_documents": rejected, "rejection_counts": dict(sorted(Counter(x["reason"] for x in rejected).items())), "groups": [{"group": g, "split": split_by_group[g], "members": [d["doc_id"] for d in docs if find(d["doc_id"]) == g]} for g in groups], "links": reasons, "split_documents": dict(sorted(Counter(d["split"] for d in output).items())), "warning": "syntactic near-duplicate audit is not a semantic paraphrase detector; unknown lineage can still leak"}

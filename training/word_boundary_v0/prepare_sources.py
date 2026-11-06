"""Acquire hash-locked public-domain excerpts; synthetic fixtures are explicit."""

import argparse
import json
from pathlib import Path
import re
import urllib.request

from .boundaries import BASE, byte_map
from .io_utils import json_bytes, jsonl_bytes, sha, write_identical_or_new


def fixture_documents():
    # All texts below were written for this prototype, not sampled from a corpus.
    cases = [
        ("english-punctuation", "en", "Dr. Reed can't re-enter the U.S. office at 3.14 p.m. See https://example.org/a?q=7."),
        ("chinese-role", "zh", "甲没有批准乙的申请。乙把17.3元交给甲。"),
        ("mixed", "mixed", "请打开https://example.org/说明，价格是USD 17.30；别把73读成37。"),
        ("unicode", "mixed", "Cafe\u0301 ≠ Café; 中文😀与𠀀、👩‍🔬一起保留。"),
        ("whitespace", "en", "  Keep\tall spaces.\r\nDo not trim.\u00a0End.  "),
        ("hyphens", "en", "The state-of-the-art tool isn't a re‑write—it's new; dogs' tags stay."),
        ("abbreviations", "en", "Mr. Fox met Mrs. Owl, e.g. near St. Lane in the U.K.; not in a URL."),
        ("numbers", "en", "Measurements: -17.30, +1,234.50, 6.02e23, 73/37 and v1.2."),
        ("symbols", "mixed", "<> & {} [] / \\ = + − × … “你好！”\u200b"),
        ("chinese-ambiguity", "zh", "研究生命起源需要谨慎。南京市长江大桥今天开放。"),
        ("chinese-condition", "zh", "如果甲没有交款，乙就不发货；除非丙先确认。"),
        ("unknown-entity", "zh", "阿澜把编号3719交给霁鸢，随后又收回了编号。"),
        ("quotes", "en", "“Wait!” she said. 'Why?' He replied: \"No.\""),
        ("url-end", "en", "Visit https://example.org/a_(b). Then email a@example.org; mail is not a URL rule."),
        ("same-prefix-a", "en", "Store the identifier 3719. Later ask its final digit."),
        ("same-prefix-b", "en", "Store the identifier 3719. Later ask its first digit."),
        ("list", "zh", "项目一：保留原文；项目二：记录来源；项目三：核对边界。"),
        ("emoji-only", "mixed", "😀✨👩‍🔬"),
        ("one-word", "en", "antidisestablishmentarianism"),
        ("empty", "mixed", ""),
        ("space-only", "mixed", " \t\r\n"),
    ]
    docs = []
    for name, lang, text in cases:
        docs.append({"doc_id": "fixture:" + name, "source_id": "self-written-fixtures-v0.1", "source_document_id": "fixture:" + name, "family_id": "fixture-family:" + ("same-prefix" if name.startswith("same-prefix") else name), "language": lang, "kind": "synthetic_fixture", "text": text, "true_document_end": True, "license": "CC0-1.0", "redistribution_allowed": True, "derived_from": []})
    base = {"source_id": "self-written-fixtures-v0.1", "language": "en", "kind": "synthetic_fixture", "true_document_end": True, "license": "CC0-1.0", "redistribution_allowed": True, "derived_from": []}
    for number in (10, 17, 73):
        docs.append({**base, "doc_id": f"fixture:inventory-{number}", "source_document_id": f"fixture:inventory-{number}", "template_family_id": "inventory-template-v1", "text": f"The red box holds {number} coins. Keep the box on the shelf."})
    docs.append({**docs[0], "doc_id": "fixture:duplicate-english", "derived_from": [docs[0]["doc_id"]]})
    docs.append({**base, "doc_id": "fixture:near-a", "source_document_id": "fixture:near-a", "text": "A careful reader keeps every original character and checks all offsets before accepting the data."})
    docs.append({**base, "doc_id": "fixture:near-b", "source_document_id": "fixture:near-b", "text": "A careful reader keeps every original character and checks all offsets before accepting the data!"})
    return docs


def prepare(cache, fetch=False):
    lock = json.loads((BASE / "sources.lock.json").read_text(encoding="utf-8"))
    documents, provenance = [], []
    for spec in lock["books"]:
        path = Path(cache) / (spec["source_id"] + ".txt")
        if not path.exists():
            if not fetch:
                raise FileNotFoundError(f"missing {path}; use --fetch to download this hash-locked source")
            request = urllib.request.Request(spec["url"], headers={"User-Agent": "ai-modeling-word-boundary-step0/0.1"})
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read(spec["size"] + 1)
            if sha(data) != spec["sha256"] or len(data) != spec["size"]:
                raise ValueError("upstream source changed; refusing unreviewed new version")
            write_identical_or_new(path, data)
        raw = path.read_bytes()
        if sha(raw) != spec["sha256"] or len(raw) != spec["size"]:
            raise ValueError("source checksum mismatch: " + spec["source_id"])
        text = raw.decode("utf-8")  # no universal-newline conversion or Unicode normalization
        start = text.index(spec["marker"])
        cursor = start
        for _ in range(spec["paragraphs"]):
            match = re.search(r"(?:\r?\n){2,}", text[cursor:])
            if match is None:
                raise ValueError("missing selected paragraph delimiter")
            cursor += match.end()
        excerpt = text[start:cursor]
        positions = byte_map(text)
        record = {**spec, "retrieved_on": lock["retrieved_on"], "source_cp": [start, cursor], "source_byte": [positions[start], positions[cursor]], "excerpt_sha256": sha(excerpt.encode("utf-8")), "normalization": "none", "license_file": f"licenses/{spec['source_id']}-license.txt"}
        provenance.append(record)
        license_match = re.search(r"THE FULL PROJECT GUTENBERG.? LICENSE", text)
        if license_match is None:
            raise ValueError("source does not contain the expected full license")
        license_start = license_match.start()
        write_identical_or_new(BASE / record["license_file"], text[license_start:].encode("utf-8"))
        documents.append({"doc_id": "real:" + spec["source_id"] + ":excerpt", "source_id": spec["source_id"], "source_document_id": spec["source_document_id"], "family_id": spec["source_document_id"], "language": spec["language"], "kind": "real_corpus_excerpt", "text": excerpt, "true_document_end": False, "license": "Public-domain-US-with-accompanying-source-notice", "redistribution_allowed": True, "source_cp": record["source_cp"], "source_byte": record["source_byte"], "derived_from": []})
    fixtures = fixture_documents()
    documents.extend(fixtures)
    provenance.append({"source_id": "self-written-fixtures-v0.1", "kind": "synthetic_fixture", "author": "AI Modeling project, examples authored for this task", "created_on": lock["retrieved_on"], "license": "CC0-1.0", "license_url": "https://creativecommons.org/publicdomain/zero/1.0/", "generator": "prepare_sources.fixture_documents", "documents": len(fixtures)})
    data = jsonl_bytes(sorted(documents, key=lambda doc: doc["doc_id"]))
    write_identical_or_new(BASE / "inputs/documents.jsonl", data)
    write_identical_or_new(BASE / "inputs/provenance.json", json_bytes({"version": lock["version"], "documents_sha256": sha(data), "sources": provenance}))
    return len(documents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("models/word-boundary-sources"))
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    print("Prepared documents:", prepare(args.cache, args.fetch))

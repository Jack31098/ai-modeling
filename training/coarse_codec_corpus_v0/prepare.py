"""Rebuild the bilingual MDN seed, check the source byte ranges and audit overlap."""

import argparse
from collections import Counter
from pathlib import Path
import re
from urllib.request import Request, urlopen

from training.word_boundary_v0.io_utils import jsonl_bytes, read_jsonl, sha
from training.word_boundary_v0.splitting import audit_normalize, shingles
from training.word_boundary_v0_2.acquire import REPOS


BASE = Path(__file__).resolve().parent
RELEASE = BASE / "mdn.jsonl"
# Manually selected half-open byte ranges in the pinned upstream Markdown.
# Each entry gives the English span followed by the Chinese span.
SELECTION = {
    "aspect_ratio": ("train", (294, 578), (259, 493)),
    "bandwidth": ("train", (222, 353), (204, 317)),
    "pixel": ("train", (99, 358), (107, 332)),
    "viewport": ("train", (226, 509), (221, 478)),
    "latency": ("dev", (195, 355), (182, 326)),
    "lossless_compression": ("test", (282, 326), (225, 258)),
    "lossy_compression": ("test", (287, 666), (270, 549)),
}
OLD_INPUTS = [
    BASE.parent / "word_boundary_v0/inputs/documents.jsonl",
    BASE.parent / "word_boundary_v0_2/sample_release/candidates.jsonl",
    BASE.parent / "generation_v1_2/local_pilot_2026_11_04/inputs/all100.jsonl",
]


def build_rows(source_dir):
    for topic, (split, en, zh) in SELECTION.items():
        for locale, (start, end) in (("en-us", en), ("zh-cn", zh)):
            lang = locale[:2]
            repo, commit = REPOS[locale]
            path = f"files/{locale}/glossary/{topic}/index.md"
            url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
            if source_dir is None:
                request = Request(url, headers={"User-Agent": "ai-modeling-mdn-seed/1"})
                with urlopen(request, timeout=20) as response:
                    raw = response.read(65537)
            else:
                raw = (source_dir / f"{locale}-{topic}.md").read_bytes()
            if not 0 <= start < end <= len(raw) <= 65536:
                raise ValueError(f"Unexpected source length at {url}")
            article = raw.decode("utf-8")
            title = re.search(r"^title: (.+)$", article, re.M).group(1)
            slug = re.search(r"^slug: (.+)$", article, re.M).group(1)
            page_locale = {"en-us": "en-US", "zh-cn": "zh-CN"}[locale]
            excerpt = raw[start:end]
            yield {
                "doc_id": f"mdn:{lang}:glossary/{topic}",
                "family_id": f"mdn:glossary/{topic}",
                "language": lang,
                "split": split,
                "text": excerpt.decode("utf-8"),
                "title": title,
                "source_url": url,
                "article_url": f"https://developer.mozilla.org/{page_locale}/docs/{slug}",
                "source_sha256": sha(raw),
                "source_byte": [start, end],
                "excerpt_sha256": sha(excerpt),
                "license": "CC-BY-SA-2.5",
                "attribution": "Mozilla Contributors",
            }


def audit(rows):
    expected = {(f"mdn:{lang}:glossary/{topic}", f"mdn:glossary/{topic}", lang, spec[0])
                for topic, spec in SELECTION.items() for lang in ("en", "zh")}
    actual = {(r["doc_id"], r["family_id"], r["language"], r["split"]) for r in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError("Missing or duplicate entry, incorrect language or split")
    previous = [r for path in OLD_INPUTS for r in read_jsonl(path)]
    for i, row in enumerate(rows):
        start, end = row["source_byte"]
        data = row["text"].encode("utf-8")
        if start < 0 or end - start != len(data) or row["excerpt_sha256"] != sha(data):
            raise ValueError("Excerpt integrity error: " + row["doc_id"])
        norm = audit_normalize(row["text"])
        grams = shingles(norm)
        references = previous + [r for r in rows[:i] if r["split"] != row["split"]]
        for old in references:
            other = audit_normalize(old["text"])
            old_grams = shingles(other)
            overlap = len(grams & old_grams) / max(1, len(grams | old_grams))
            shorter, longer = sorted((norm, other), key=len)
            same_family = row["family_id"] == old.get("family_id")
            if (same_family or norm == other or overlap >= 0.85 or
                    (len(shorter) >= 24 and shorter in longer)):
                raise ValueError(f"Overlap: {row['doc_id']} / {old['doc_id']}")
    print("Historical inputs checked:", len(previous))
    print("Rows by split:", dict(Counter(r["split"] for r in rows)))
    print("Code points:", sum(len(r["text"]) for r in rows))
    print("UTF-8 bytes:", sum(len(r["text"].encode("utf-8")) for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--upstream", action="store_true", help="refetch the 14 pinned files")
    group.add_argument("--source-dir", type=Path, help="use cached <locale>-<topic>.md files")
    args = parser.parse_args()
    if args.upstream or args.source_dir:
        rebuilt = list(build_rows(args.source_dir))
        audit(rebuilt)
        if RELEASE.exists():
            if rebuilt != read_jsonl(RELEASE):
                raise ValueError("Rebuilt records disagree with the committed release")
        else:
            RELEASE.write_bytes(jsonl_bytes(rebuilt))
        print("Verified all upstream slices and source hashes.")
    else:
        audit(read_jsonl(RELEASE))


if __name__ == "__main__":
    main()

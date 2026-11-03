"""Lossless units; reference segmentation is deliberately separate from online policy."""

from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import unicodedata

BASE = Path(__file__).parent
RULES = json.loads((BASE / "rules.json").read_text(encoding="utf-8"))
NUMBER = re.compile(r"[+-]?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
INITIALISM = re.compile(r"(?:[A-Za-z]\.){2,}")
URL = re.compile(r"(?:https?://|www\.)[^\s<>\"“”，。；：！？]+", re.I)
ABBREV = re.compile("(?:" + "|".join(re.escape(x) for x in sorted(RULES["english"]["abbreviations"], key=lambda x: (-len(x), x))) + ")", re.I)


def is_han(char):
    code = ord(char)
    return 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF or 0x20000 <= code <= 0x323AF


def is_letter(char):
    return not is_han(char) and unicodedata.category(char).startswith("L")


def byte_map(text):
    result = [0]
    for char in text:
        result.append(result[-1] + len(char.encode("utf-8")))
    return result


@lru_cache(maxsize=1)
def chinese_segmenter():
    import jieba

    if importlib.metadata.version("jieba") != "0.42.1":
        raise ValueError("jieba version must be 0.42.1")
    lock = json.loads((BASE / "sources.lock.json").read_text(encoding="utf-8"))
    dictionary = Path(jieba.__file__).parent / "dict.txt"
    actual = hashlib.sha256(dictionary.read_bytes()).hexdigest()
    if actual != lock["jieba"]["dictionary_sha256"]:
        raise ValueError("jieba dictionary checksum mismatch")
    jieba.setLogLevel(40)
    segmenter = jieba.Tokenizer(str(dictionary))
    # Avoid global jieba state, user dictionaries, parallel mode and external cache writes.
    segmenter.tmp_dir = str(dictionary.parent)
    segmenter.cache_file = "word-boundary-v0.cache"
    segmenter.initialize()
    return segmenter


def dependency_info():
    import jieba

    chinese_segmenter()
    dictionary = Path(jieba.__file__).parent / "dict.txt"
    return {"jieba": importlib.metadata.version("jieba"), "dictionary_sha256": hashlib.sha256(dictionary.read_bytes()).hexdigest(), "HMM": False, "user_dictionary": None, "unicode_database": unicodedata.unidata_version}


def segment(text):
    """All whitespace/punctuation/symbols retained, [start,end) code point and byte spans."""
    offsets = byte_map(text)
    units = []

    def add(start, end, kind):
        value = text[start:end]
        units.append({"text": value, "cp": [start, end], "byte": [offsets[start], offsets[end]], "kind": kind, "sentence_terminal_candidate": kind == "punct" and value in ".!?。！？"})

    i = 0
    while i < len(text):
        start = i
        char = text[i]
        if char.isspace():
            i += 1
            while i < len(text) and text[i].isspace():
                i += 1
            add(start, i, "space")
            continue
        if is_han(char):
            i += 1
            while i < len(text) and is_han(text[i]):
                i += 1
            cursor = start
            for word in chinese_segmenter().cut(text[start:i], cut_all=False, HMM=False):
                add(cursor, cursor + len(word), "han_word")
                cursor += len(word)
            if cursor != i:
                raise ValueError("Chinese segmentation lost coverage")
            continue
        match = URL.match(text, i)
        if match:
            end = match.end()
            while end > i and text[end - 1] in RULES["english"]["url_trailing_punctuation"]:
                end -= 1
            if end > i:
                add(i, end, "url")
                i = end
                continue
        match = NUMBER.match(text, i)
        if match:
            i = match.end()
            add(start, i, "number")
            continue
        match = INITIALISM.match(text, i) or ABBREV.match(text, i)
        if match and (match.end() == len(text) or not is_letter(text[match.end()])):
            i = match.end()
            add(start, i, "word")
            continue
        if is_letter(char):
            i += 1
            while i < len(text):
                if is_letter(text[i]) or unicodedata.category(text[i]).startswith("M"):
                    i += 1
                elif text[i] in RULES["english"]["internal_apostrophes"] + RULES["english"]["internal_hyphens"] and i + 1 < len(text) and is_letter(text[i + 1]):
                    i += 1
                else:
                    break
            add(start, i, "word")
            continue
        i += 1
        # Combining marks stay with a preceding symbol too; do not normalize them.
        while i < len(text) and unicodedata.category(text[i]).startswith("M"):
            i += 1
        add(start, i, "punct" if unicodedata.category(char).startswith("P") else "symbol")
    validate_units(text, units)
    return units


def validate_units(text, units):
    offsets = byte_map(text)
    cursor = 0
    for unit in units:
        a, b = unit["cp"]
        if a != cursor or b <= a or unit["text"] != text[a:b] or unit["byte"] != [offsets[a], offsets[b]]:
            raise ValueError("non-partitioning or misaligned units")
        cursor = b
    if cursor != len(text) or "".join(u["text"] for u in units) != text:
        raise ValueError("incomplete reconstruction")


def online_english_chunks(prefix, true_document_end=False):
    """Prefix-only buffered decisions; Han-containing chunks intentionally abstain."""
    decisions = []
    for match in re.finditer(r"\S+", prefix):
        end = match.end()
        if end == len(prefix) and not true_document_end:
            continue
        chunk = match.group()
        if any(is_han(c) for c in chunk):
            continue
        # A real EOF is a received event at n+1, not an implicit lookahead at n.
        available = end + 1
        units = segment(chunk)
        decisions.append({"start": match.start(), "end": end, "available_at": available, "boundaries": [match.start() + u["cp"][1] for u in units]})
    return decisions

"""Bounded prefix-only policy. Fixed lexicon, no global DAG or offline labels."""

from functools import lru_cache
import importlib.metadata
import json
from pathlib import Path
import unicodedata

from training.word_boundary_v0.boundaries import is_han, segment
from training.word_boundary_v0.io_utils import sha

BASE = Path(__file__).parent
POLICY = json.loads((BASE / "policy.json").read_text(encoding="utf-8"))
POLICY_SHA = sha((BASE / "policy.json").read_bytes())


@lru_cache(maxsize=1)
def lexicon():
    import jieba
    if importlib.metadata.version("jieba") != "0.42.1":
        raise ValueError("dictionary package version drift")
    raw = (Path(jieba.__file__).parent / "dict.txt").read_bytes()
    if sha(raw) != POLICY["dictionary_sha256"]:
        raise ValueError("dictionary hash drift")
    words = frozenset(line.split()[0] for line in raw.decode("utf-8").splitlines()
                      if line.split() and 1 <= len(line.split()[0]) <= POLICY["han_max_word_cp"]
                      and all(is_han(c) for c in line.split()[0]))
    return words, sha("\n".join(sorted(words)).encode("utf-8"))


def mode_of(char):
    if is_han(char):
        return "han"
    # ASCII punctuation stays in capped chunks for URLs, decimals and abbreviations.
    if char.isspace() or ord(char) > 127 and unicodedata.category(char)[0] in "PS":
        return "separator"
    return "non_han"


class Stream:
    def __init__(self):
        _, self.lexicon_sha = lexicon()
        self.pending = ""
        self.mode = None
        self.cp = self.byte = self.clock = 0
        self.start_cp = self.start_byte = 0
        self.ended = False
        self.max_pending_cp = self.max_pending_byte = self.max_delay_steps = 0

    def _emit(self, count, reason, kind, forced=False):
        value = self.pending[:count]
        end_cp = self.start_cp + count
        end_byte = self.start_byte + len(value.encode("utf-8"))
        # Age since first character of this unit, including any real EOF event.
        self.max_delay_steps = max(self.max_delay_steps, self.clock - self.start_cp - 1)
        record = {"text": value, "cp": [self.start_cp, end_cp], "byte": [self.start_byte, end_byte],
                  "kind": kind, "decision_at": self.clock, "available_at": self.clock,
                  "evidence_available_at": self.clock, "observation_cutoff": self.clock,
                  "status": "committed", "commit_reason": reason, "continuation_possible": forced,
                  "action_source": "prefix-fmm-reference-policy", "main_input_allowed": False}
        self.pending = self.pending[count:]
        self.start_cp, self.start_byte = end_cp, end_byte
        return record

    def _han(self, reason, flush=False):
        words, _ = lexicon()
        limit = POLICY["han_max_word_cp"]
        while self.pending and (flush or len(self.pending) >= limit):
            length = next((n for n in range(min(limit, len(self.pending)), 0, -1)
                           if self.pending[:n] in words), 1)
            kind = "han_lexicon" if self.pending[:length] in words else "han_unknown"
            yield self._emit(length, reason, kind)

    def _flush(self, reason, forced=False):
        if self.mode == "han":
            yield from self._han(reason, flush=True)
        elif self.pending:
            assert not any(is_han(c) for c in self.pending)
            # Only segment non-Han text: this cannot consult the offline Han DAG.
            for unit in segment(self.pending):
                yield self._emit(len(unit["text"]), reason, unit["kind"], forced)
        self.mode = None

    def feed(self, chunk):
        if self.ended:
            raise ValueError("cannot feed after true EOF")
        for char in chunk:
            self.clock += 1
            mode = mode_of(char)
            if self.pending and (mode != self.mode or mode == "separator"):
                yield from self._flush("observed_terminator")
            if not self.pending:
                self.start_cp, self.start_byte = self.cp, self.byte
            self.mode = mode
            self.pending += char
            self.cp += 1
            self.byte += len(char.encode("utf-8"))
            self.max_pending_cp = max(self.max_pending_cp, len(self.pending))
            self.max_pending_byte = max(self.max_pending_byte, len(self.pending.encode("utf-8")))
            if mode == "han":
                yield from self._han("six_visible_han")
            elif mode == "separator":
                yield self._emit(1, "observed_separator", "space" if char.isspace() else "punct_or_symbol")
                self.mode = None
            else:
                is_url = self.pending.lower().startswith(("http://", "https://", "www."))
                cap = POLICY["url_buffer_cap_cp"] if is_url else POLICY["non_han_buffer_cap_cp"]
                if len(self.pending) >= cap:
                    yield from self._flush("budget_cap", forced=True)

    def pause(self):
        """Window/excerpt end is not evidence of a word boundary or true EOF."""
        return self.snapshot()

    def finish(self):
        if self.ended:
            raise ValueError("duplicate EOF")
        self.clock += 1
        yield from self._flush("true_eof")
        self.ended = True

    def snapshot(self):
        return {"policy_sha256": POLICY_SHA, "lexicon_sha256": self.lexicon_sha,
                **{name: getattr(self, name) for name in ("pending", "mode", "cp", "byte", "clock",
                   "start_cp", "start_byte", "ended", "max_pending_cp", "max_pending_byte", "max_delay_steps")}}

    @classmethod
    def restore(cls, state):
        obj = cls()
        expected = obj.snapshot()
        if set(state) != set(expected) or state["policy_sha256"] != POLICY_SHA or state["lexicon_sha256"] != obj.lexicon_sha:
            raise ValueError("incompatible state")
        for name in expected:
            if name not in ("policy_sha256", "lexicon_sha256"):
                setattr(obj, name, state[name])
        numeric = ("cp", "byte", "clock", "start_cp", "start_byte", "max_pending_cp", "max_pending_byte", "max_delay_steps")
        if (any(type(getattr(obj, n)) is not int or getattr(obj, n) < 0 for n in numeric)
                or type(obj.ended) is not bool or not isinstance(obj.pending, str)):
            raise ValueError("invalid state types or counters")
        cap = 64 if obj.pending.lower().startswith(("http://", "https://", "www.")) else 32
        if (obj.cp - obj.start_cp != len(obj.pending) or obj.byte - obj.start_byte != len(obj.pending.encode("utf-8"))
                or len(obj.pending) >= cap or obj.clock != obj.cp + int(obj.ended)
                or obj.mode not in (None, "han", "non_han", "separator")
                or obj.pending and (obj.mode is None or any(mode_of(c) != obj.mode for c in obj.pending))
                or obj.mode == "han" and len(obj.pending) >= POLICY["han_max_word_cp"]
                or obj.max_pending_cp > 64 or obj.max_pending_byte > 256 or obj.max_delay_steps > 64
                or obj.ended and obj.pending):
            raise ValueError("invalid state geometry")
        return obj


def run(text, true_eof=False, chunk_width=17):
    stream = Stream()
    commits = []
    for start in range(0, len(text), chunk_width):
        commits.extend(stream.feed(text[start:start + chunk_width]))
    if true_eof:
        commits.extend(stream.finish())
    return commits, stream.snapshot()

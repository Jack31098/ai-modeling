from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from training.word_boundary_v0.boundaries import byte_map
from training.word_boundary_v0.io_utils import sha
from training.word_boundary_v0_2.build import baseline_errors, build
from training.word_boundary_v0_2.candidates import BASE, automatic_suggestion, duplicate_audit, review_template, source_candidates, validate_review
from training.word_boundary_v0_2.streaming import Stream, lexicon, run


def partition(test, text, commits, state):
    offsets, cursor = byte_map(text), 0
    for unit in commits:
        a, b = unit["cp"]
        test.assertEqual(a, cursor)
        test.assertEqual(unit["text"], text[a:b])
        test.assertEqual(unit["byte"], [offsets[a], offsets[b]])
        test.assertGreaterEqual(unit["decision_at"], b)
        test.assertLessEqual(unit["decision_at"] - a - 1, 64)
        test.assertEqual(unit["observation_cutoff"], unit["evidence_available_at"])
        cursor = b
    test.assertEqual(state["pending"], text[cursor:])
    test.assertEqual("".join(u["text"] for u in commits) + state["pending"], text)


class IncrementTests(unittest.TestCase):
    def test_exact_source_spans_and_rights(self):
        docs = source_candidates()
        self.assertEqual(len(docs), 4)
        self.assertEqual(len({d["family_id"] for d in docs}), 2)
        for d in docs:
            raw = (BASE / d["source"]["local_path"]).read_bytes()
            a, b = d["source_byte"]
            self.assertEqual(raw[a:b].decode("utf-8"), d["text"])
            self.assertEqual(sha(raw[a:b]), d["excerpt_sha256"])
            self.assertEqual(d["license"], "CC-BY-SA-2.5")
            self.assertFalse(d["training_eligible"])
            self.assertIsNone(d["split"])
            if d["language"] == "zh":
                self.assertIsNotNone(d["translation_source_commit"])

    def test_overlap_audit_includes_all_old_train_and_inputs(self):
        docs = source_candidates()
        report = duplicate_audit(docs)
        self.assertEqual((report["old_input_count"], report["old_train_count"]), (29, 21))
        self.assertTrue(all(not x["matches"] for x in report["results"]))
        from training.word_boundary_v0.io_utils import read_jsonl
        from training.word_boundary_v0_2.candidates import OLD
        fake = deepcopy(docs[0])
        fake["text"] = read_jsonl(OLD / "inputs/documents.jsonl")[0]["text"]
        self.assertTrue(duplicate_audit([fake])["results"][0]["matches"])

    def test_reviews_start_empty_and_reject_unreviewed(self):
        d = source_candidates()[0]
        r = review_template(d)
        self.assertEqual(r["units"], [])
        with self.assertRaises(ValueError):
            validate_review(d, r)
        r["review_status"] = "accepted"
        with self.assertRaises(ValueError):
            validate_review(d, r)
        r["annotator"] = "synthetic-validator-test-only"
        r["units"] = automatic_suggestion(d)["units"]
        self.assertTrue(validate_review(d, r))
        r["units"][0]["byte"][0] = 1
        with self.assertRaises(ValueError):
            validate_review(d, r)

    def test_unresolved_ambiguity_rejected(self):
        d = source_candidates()[0]
        r = review_template(d)
        r.update(review_status="accepted", annotator="synthetic-validator-test-only", units=automatic_suggestion(d)["units"],
                 ambiguities=[{"span_cp": [0, 1], "resolution": None}])
        with self.assertRaises(ValueError):
            validate_review(d, r)

    def test_unicode_unknown_and_whitespace_partition(self):
        for text in ("", " \t\r\n", "甲没有批准乙的申请。", "研究生命起源。", "𠀀𠀀𠀀𠀀𠀀𠀀。", "Cafe\u0301 ≠ Café; 👩‍🔬", "请看https://example.org/说明，价格17.30。"):
            for eof in (False, True):
                commits, state = run(text, eof)
                partition(self, text, commits, state)
                if eof:
                    self.assertEqual(state["pending"], "")

    def test_prefix_future_counterfactuals(self):
        prefix = "研究生命起源需要谨慎。请访问https://example.org"
        for future in ("/a?q=7。后文", " 更多英文 words", "说明𠀀", ""):
            before, state = run(prefix, False)
            after, _ = run(prefix + future, True)
            self.assertEqual(before, [x for x in after if x["decision_at"] <= len(prefix)])
            self.assertFalse(state["ended"])

    def test_transport_chunking_invariant(self):
        text = "研究生命起源。Dr. Reed can't re-enter https://example.org/说明。\r\n" * 5
        expected = run(text, True, 10000)
        for width in (1, 2, 3, 7, 32, 64, 128):
            self.assertEqual(expected, run(text, True, width))

    def test_arbitrary_chunks_and_fuzz(self):
        rng = random.Random(1829)
        for _ in range(40):
            text = "".join(rng.choice("研究生命阿澜𠀀😀a01_':/\u0301 \t\r\n") for _ in range(100))
            expected = run(text, True)
            stream, actual, cursor = Stream(), [], 0
            while cursor < len(text):
                width = rng.randint(1, 19)
                actual.extend(stream.feed(text[cursor:cursor + width]))
                cursor += width
            actual.extend(stream.finish())
            self.assertEqual(expected, (actual, stream.snapshot()))
            partition(self, text, actual, stream.snapshot())

    def test_actual_eof_n_plus_one(self):
        text = "antidisestablishmentarianism"
        commits, state = run(text, False)
        self.assertEqual(commits, [])
        self.assertEqual(state["clock"], 28)
        stream = Stream.restore(state)
        self.assertEqual(stream.pause(), state)
        eof = list(stream.finish())
        self.assertEqual(eof[0]["decision_at"], 29)
        with self.assertRaises(ValueError):
            list(stream.feed("x"))
        with self.assertRaises(ValueError):
            list(stream.finish())

    def test_excerpt_window_state_resume(self):
        text = "研究生命起源需要谨慎。https://example.org" + "a" * 80 + " 甲没有批准乙。"
        for width in (1, 5, 13, 64):
            stream, actual = Stream(), []
            for a in range(0, len(text), width):
                actual.extend(stream.feed(text[a:a + width]))
                before = stream.snapshot()
                self.assertEqual(stream.pause(), before)
                stream = Stream.restore(json.loads(json.dumps(before)))
            actual.extend(stream.finish())
            self.assertEqual((actual, stream.snapshot()), run(text, True))

    def test_bounded_long_runs_and_delays(self):
        # Consume output into a constant-size counter: output storage is not the buffer.
        for chunk in ("研究生命起源", "𠀀" * 8, "x" * 17, "https://" + "a" * 90):
            stream, count = Stream(), 0
            for _ in range(1200):
                count += sum(len(x["text"]) for x in stream.feed(chunk))
                state = stream.snapshot()
                self.assertLessEqual(len(state["pending"]), 63)
                self.assertNotIn("history", state)
                self.assertNotIn("committed", state)
            count += sum(len(x["text"]) for x in stream.finish())
            self.assertEqual(count, len(chunk) * 1200)
            self.assertLessEqual(stream.max_pending_cp, 64)
            self.assertLessEqual(stream.max_pending_byte, 256)
            self.assertLessEqual(stream.max_delay_steps, 64)
            if all(ord(c) > 127 for c in chunk):
                self.assertLessEqual(stream.max_pending_cp, 6)

    def test_budget_caps_explicit(self):
        commits, state = run("https://" + "a" * 100, True)
        forced = [x for x in commits if x["commit_reason"] == "budget_cap"]
        self.assertTrue(forced)
        self.assertTrue(all(x["continuation_possible"] for x in forced))
        self.assertEqual(state["max_pending_cp"], 64)

    def test_no_offline_or_label_oracle(self):
        text = "研究生命起源。甲没有批准乙的申请。"
        expected = run(text, True)
        with patch("training.word_boundary_v0.boundaries.chinese_segmenter", side_effect=AssertionError("offline oracle")):
            self.assertEqual(expected, run(text, True))
        self.assertEqual(set(Stream().snapshot()) & {"labels", "future_answer", "gold"}, set())

    def test_invalid_or_drifting_state_rejected(self):
        _, state = run("研究生", False)
        for key, value in (("policy_sha256", "x"), ("lexicon_sha256", "x"), ("byte", 999)):
            bad = {**state, key: value}
            with self.assertRaises(ValueError):
                Stream.restore(bad)

    def test_reference_is_not_gold(self):
        commits, _ = run("研究生命起源。", True)
        self.assertEqual(commits[0]["text"], "研究生")
        self.assertEqual(commits[0]["decision_at"], 6)
        self.assertNotIn("gold_accuracy", commits[0])

    def test_deterministic_release_and_preserved_baseline(self):
        self.assertEqual(baseline_errors(), [])
        temp_root = BASE.parents[1] / "models"
        with tempfile.TemporaryDirectory(prefix="wb-v02-test-", dir=temp_root) as tmp:
            if not Path(tmp).resolve().is_relative_to(temp_root.resolve()) or temp_root.resolve() != BASE.parents[1].resolve() / "models":
                raise ValueError("temporary cleanup target outside intended workspace")
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            manifest = build(a)
            self.assertEqual(manifest["policy_sha256"], sha((a / "policy.json").read_bytes()))
            self.assertTrue((a / "notices/v0/licenses/jieba-MIT.txt").exists())
            build(b)
            left = {p.relative_to(a).as_posix(): p.read_bytes() for p in a.rglob("*") if p.is_file()}
            right = {p.relative_to(b).as_posix(): p.read_bytes() for p in b.rglob("*") if p.is_file()}
            self.assertEqual(left, right)
            with self.assertRaises(FileExistsError):
                build(a)
            c = Path(tmp) / "c"
            with patch("training.word_boundary_v0_2.build.automatic_suggestion",
                       side_effect=lambda doc: {"doc_id": doc["doc_id"], "units": [], "deliberately_wrong_future_label": True}):
                build(c)
            for folder in ("commits", "observations", "states"):
                for file in (a / folder).glob("*"):
                    self.assertEqual(file.read_bytes(), (c / folder / file.name).read_bytes())
        self.assertEqual(baseline_errors(), [])

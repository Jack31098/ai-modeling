from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest
import unicodedata

from training.word_boundary_v0.alignment import align_tokens, codepoint_fixture
from training.word_boundary_v0.boundaries import BASE, byte_map, dependency_info, online_english_chunks, segment, validate_units
from training.word_boundary_v0.build import build
from training.word_boundary_v0.events import compile_tapes, consume_labels, model_input, slice_document, training_pairs
from training.word_boundary_v0.io_utils import read_jsonl, sha
from training.word_boundary_v0.prepare_sources import fixture_documents
from training.word_boundary_v0.splitting import assign_documents


def doc(name, text, **kwargs):
    return {"doc_id": name, "source_id": "self-written-test", "source_document_id": name, "text": text, "kind": "synthetic_fixture", "language": "mixed", "redistribution_allowed": True, "true_document_end": True, "content_sha256": sha(text.encode()), **kwargs}


class BoundaryTests(unittest.TestCase):
    def test_english_policy(self):
        text = "Dr. Reed can't re-enter 3.14 https://example.org/a."
        self.assertEqual([u["text"] for u in segment(text)], ["Dr.", " ", "Reed", " ", "can't", " ", "re-enter", " ", "3.14", " ", "https://example.org/a", "."])

    def test_abbreviation_sentence_end(self):
        units = segment("U.S. policy. Yes!")
        self.assertEqual(sum(u["sentence_terminal_candidate"] for u in units), 2)
        self.assertEqual(units[0]["text"], "U.S.")

    def test_chinese_dictionary_regression(self):
        self.assertEqual([u["text"] for u in segment("研究生命起源。")], ["研究", "生命", "起源", "。"])

    def test_chinese_punctuation_terminates_url(self):
        text = "打开https://example.org/说明，继续。"
        units = segment(text)
        self.assertEqual([u["text"] for u in units if u["kind"] == "url"], ["https://example.org/说明"])
        self.assertIn("，", [u["text"] for u in units])
        validate_units(text, units)

    def test_mixed_unicode_and_line_endings(self):
        for text in ("", " \t\r\n", "中文𠀀😀", "Cafe\u0301 ≠ Café", "请看https://example.org/说明。", "👩‍🔬\u200b"):
            with self.subTest(text=text):
                units = segment(text)
                validate_units(text, units)
                encoded = text.encode()
                self.assertEqual(b"".join(encoded[u["byte"][0]:u["byte"][1]] for u in units), encoded)

    def test_fuzz_partition(self):
        rng = random.Random(7391)
        for _ in range(100):
            text = "".join(rng.choice("ab'’-‐.012中甲研 \t\n\r😀\u0301") for _ in range(45))
            validate_units(text, segment(text))

    def test_coverage_rejects_overlap_and_gap(self):
        units = segment("ab cd")
        units[1]["cp"][0] = 1
        with self.assertRaises(ValueError):
            validate_units("ab cd", units)

    def test_frozen_dictionary(self):
        info = dependency_info()
        self.assertEqual(info["jieba"], "0.42.1")
        self.assertEqual(info["dictionary_sha256"], "7197c3211ddd98962b036cdf40324d1ea2bfaa12bd028e68faa70111a88e12a8")
        self.assertFalse(info["HMM"])


class AlignmentTests(unittest.TestCase):
    def align(self, text, tokens, **kwargs):
        return align_tokens(text, [u["cp"][1] for u in segment(text)], tokens, offset_unit=kwargs.pop("offset_unit", "unicode_code_point"), tokenizer_id="test-fixture", **kwargs)

    def test_identity_and_special_tokens(self):
        text = "中😀 hi!"
        tokens = [{"id": "BOS", "special": True, "span": [0, 0]}] + codepoint_fixture(text) + [{"id": "EOS", "special": True, "span": None}]
        result = self.align(text, tokens)
        self.assertTrue(result["usable_without_conflicts"])
        self.assertEqual(len(result["notes"]), 2)

    def test_multi_boundaries_in_one_token_not_snapped(self):
        text = "red blue!"
        result = self.align(text, [{"id": 1, "span": [0, len(text)]}])
        self.assertFalse(result["usable_without_conflicts"])
        self.assertIn("multiple_boundaries_inside_token", {e["reason"] for e in result["errors"]})
        self.assertEqual(result["boundaries"][0]["cp"], 3)
        self.assertEqual(result["boundaries"][0]["status"], "boundary_inside_token")
        self.assertFalse(result["boundaries"][0]["loss_mask"])

    def test_utf8_adapter(self):
        text = "甲😀e\u0301"
        offsets = byte_map(text)
        tokens = [{"id": i, "span": [offsets[i], offsets[i + 1]], "surface": c} for i, c in enumerate(text)]
        self.assertTrue(self.align(text, tokens, offset_unit="utf8_byte")["usable_without_conflicts"])

    def test_byte_offset_inside_multibyte_character(self):
        result = self.align("中", [{"id": 1, "span": [0, 1]}, {"id": 2, "span": [1, 3]}], offset_unit="utf8_byte")
        self.assertFalse(result["usable_without_conflicts"])
        self.assertIn("unicode_or_out_of_range_offset", {e["reason"] for e in result["errors"]})

    def test_utf16_and_broken_surrogate(self):
        self.assertTrue(self.align("😀甲", [{"id": 1, "span": [0, 2]}, {"id": 2, "span": [2, 3]}], offset_unit="utf16_code_unit")["usable_without_conflicts"])
        self.assertFalse(self.align("😀", [{"id": 1, "span": [0, 1]}], offset_unit="utf16_code_unit")["usable_without_conflicts"])

    def test_normalization_mismatch(self):
        text = "Cafe\u0301"
        result = self.align(text, codepoint_fixture(text), original_text=unicodedata.normalize("NFC", text))
        self.assertFalse(result["usable_without_conflicts"])
        self.assertFalse(any(b["loss_mask"] for b in result["boundaries"]))

    def test_surface_mismatch_and_coverage(self):
        for tokens, reason in [([{ "id": 1, "span": [0, 2], "surface": "XY"}], "surface_mismatch"), ([{"id": 1, "span": [1, 2]}], "token_coverage_gap"), ([{"id": 1, "span": [0, 2]}, {"id": 2, "span": [1, 2]}], "token_overlap_or_order"), ([{"id": "special", "span": [0, 2], "special": True}], "special_token_has_text_span"), ([{"id": 1, "span": [0, 0]}], "empty_or_reversed_offset")]:
            with self.subTest(reason=reason):
                result = self.align("ab", tokens)
                self.assertFalse(result["usable_without_conflicts"])
                self.assertIn(reason, {e["reason"] for e in result["errors"]})

    def test_unknown_coordinates_fail_closed(self):
        with self.assertRaises(ValueError):
            self.align("abc", [], offset_unit="unspecified")


class GroupTests(unittest.TestCase):
    def test_exact_near_template_and_lineage(self):
        docs = [doc(str(i), "Unique document " + chr(65 + i) * 30) for i in range(15)]
        docs += [doc("base", "The red box has 17 coins and belongs on the upper shelf."), doc("derived", "The red box has 73 coins and belongs on the upper shelf."), doc("duplicate", docs[0]["text"]), doc("near-a", "A reader must retain all characters and verify the original offsets carefully."), doc("near-b", "A reader must retain all characters and verify the original offsets carefully!"), doc("lineage", "Different text in the same derived family.", derived_from=["base"])]
        result, audit = assign_documents(docs)
        lookup = {d["doc_id"]: d for d in result}
        self.assertEqual(lookup["base"]["isolation_group"], lookup["derived"]["isolation_group"])
        self.assertEqual(lookup["base"]["split"], lookup["lineage"]["split"])
        self.assertEqual(lookup["near-a"]["isolation_group"], lookup["near-b"]["isolation_group"])
        self.assertEqual(audit["rejection_counts"]["exact_duplicate"], 1)
        self.assertIn("near_duplicate", {link["reason"] for link in audit["links"]})
        self.assertIn("number_template", {link["reason"] for link in audit["links"]})
        self.assertEqual({d["split"] for d in result}, {"train", "dev", "test"})

    def test_work_and_parallel_family_isolation(self):
        docs = [doc("a", "English one", source_document_id="book"), doc("b", "English two", source_document_id="book"), doc("c", "一个翻译", parallel_family_id="translation"), doc("d", "Another translation", parallel_family_id="translation")]
        result, _ = assign_documents(docs)
        self.assertEqual(result[0]["isolation_group"], result[1]["isolation_group"])
        self.assertEqual(result[2]["isolation_group"], result[3]["isolation_group"])

    def test_order_independent_determinism(self):
        docs = fixture_documents()
        self.assertEqual(assign_documents(docs), assign_documents(list(reversed(docs))))

    def test_empty_rights_and_missing_lineage(self):
        _, audit = assign_documents([doc("empty", ""), doc("space", " \n"), doc("rights", "rights unknown", redistribution_allowed=False)])
        self.assertEqual(audit["kept_documents"], 0)
        self.assertEqual(audit["rejection_counts"], {"empty_or_whitespace_only": 2, "unverified_redistribution": 1})
        with self.assertRaises(ValueError):
            assign_documents([doc("missing", "some text", derived_from=["not-in-input"])])

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            assign_documents([doc("same", "abc"), doc("same", "def")])


class VisibilityTests(unittest.TestCase):
    def test_english_prefix_consistency_and_lookahead(self):
        for text in ("can't re-enter now.", "Visit https://example.org/x?q=3. Done.", "Dr. Fox waits.", "英文 English and 中文。"):
            whole = online_english_chunks(text, True)
            for cutoff in range(len(text) + 1):
                expected = [d for d in whole if d["available_at"] <= cutoff]
                self.assertEqual(online_english_chunks(text[:cutoff]), expected)
        self.assertEqual(online_english_chunks("can't"), [])
        self.assertEqual(online_english_chunks("can't ")[0]["available_at"], 6)

    def test_chinese_reference_is_not_stream_action(self):
        d = doc("zh", "研究生命起源。")
        observations, labels = compile_tapes(d, segment(d["text"]))
        self.assertEqual(consume_labels(labels, 3, "READ_FULL"), [])
        self.assertEqual(consume_labels(labels, len(d["text"]) + 1, "READ_STREAM"), [])
        self.assertFalse(labels[0]["allow_stream_action_forcing"])
        self.assertTrue(next(training_pairs(observations, labels, "READ_FULL"))["model_input"]["input_complete"])

    def test_no_labels_or_future_in_model_input(self):
        a = doc("a", "Keep 3719. Later use red.")
        b = doc("b", "Keep 3719. Later use blue and more text.")
        oa, la = compile_tapes(a, segment(a["text"]))
        ob, _ = compile_tapes(b, segment(b["text"]))
        cutoff = len("Keep 3719.")
        self.assertEqual(model_input(oa, cutoff), model_input(ob, cutoff))
        mutated = deepcopy(oa)
        for event in mutated:
            event.update({"target": "SECRET", "future_mask": [1, 1, 1], "reference_latent": [123]})
        before = model_input(oa, cutoff)
        la[0]["target"] = "CHANGED LABEL"
        self.assertEqual(model_input(mutated, cutoff), before)
        self.assertNotIn("SECRET", json.dumps(model_input(mutated, cutoff)))
        self.assertEqual(set(before), {"text", "input_complete", "document_ended"})

    def test_bad_supervision_timing_rejected(self):
        d = doc("bad", "One word.")
        _, labels = compile_tapes(d, segment(d["text"]))
        label = labels[0]
        label["evidence_available_at"] = 9999
        with self.assertRaises(ValueError):
            consume_labels(labels, 100, label["mode"])

    def test_slicing_no_synthetic_eos_or_cross_split(self):
        d = doc("slice", "alpha beta gamma delta", split="test", isolation_group="work", true_document_end=False)
        units = segment(d["text"])
        _, labels = compile_tapes(d, units)
        slices = slice_document(d, units, labels, 7)
        self.assertEqual("".join(s["text"] for s in slices), d["text"])
        self.assertTrue(all(s["split"] == "test" and not s["inject_eos"] for s in slices))
        self.assertTrue(slices[-1]["is_artificial_window_end"])
        self.assertTrue(any(s["crossing_units_masked"] for s in slices))
        self.assertFalse(any(label["buffer_cp"][1] == len(d["text"]) for label in labels if label["mode"] == "READ_STREAM"))

    def test_true_eof_is_separate_observed_event(self):
        d = doc("end", "last")
        observations, labels = compile_tapes(d, segment(d["text"]))
        self.assertFalse(model_input(observations, 4)["document_ended"])
        self.assertTrue(model_input(observations, 5)["document_ended"])
        self.assertEqual(consume_labels(labels, 4, "READ_STREAM"), [])
        self.assertEqual(consume_labels(labels, 5, "READ_STREAM")[0]["available_at"], 5)

    def test_eof_versus_same_prefix_continuation_and_label_free_clock(self):
        short, continued = doc("short", "last"), doc("long", "lastly continues")
        obs_a, labels_a = compile_tapes(short, segment(short["text"]))
        obs_b, labels_b = compile_tapes(continued, segment(continued["text"]))
        self.assertEqual(model_input(obs_a, 4), model_input(obs_b, 4))
        self.assertEqual(consume_labels(labels_a, 4, "READ_STREAM"), [])
        self.assertEqual(consume_labels(labels_b, 4, "READ_STREAM"), [])
        frames_a = list(training_pairs(obs_a, labels_a, "READ_STREAM"))
        self.assertTrue(frames_a[-1]["model_input"]["document_ended"])
        self.assertEqual(frames_a[-1]["decision_at"], 5)
        self.assertTrue(frames_a[-1]["loss_targets"])
        self.assertFalse(model_input(obs_b, 5)["document_ended"])
        self.assertEqual(consume_labels(labels_b, 5, "READ_STREAM"), [])
        self.assertEqual([p["model_input"] for p in frames_a], [p["model_input"] for p in training_pairs(obs_a, [], "READ_STREAM")])


class ReleaseTests(unittest.TestCase):
    def test_provenance_and_input_integrity(self):
        provenance = json.loads((BASE / "inputs/provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(sha((BASE / "inputs/documents.jsonl").read_bytes()), provenance["documents_sha256"])
        docs = read_jsonl(BASE / "inputs/documents.jsonl")
        self.assertEqual(sum(d["kind"] == "real_corpus_excerpt" for d in docs), 2)
        for source in provenance["sources"]:
            if "excerpt_sha256" in source:
                selected = next(d for d in docs if d["source_id"] == source["source_id"])
                self.assertEqual(sha(selected["text"].encode()), source["excerpt_sha256"])

    def test_end_to_end_deterministic_and_manifest(self):
        temp_root = (BASE.parents[1] / "models/word-boundary-test-tmp").resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
            # TemporaryDirectory recursively removes only this newly created, verified target.
            self.assertTrue(Path(tmp).resolve().is_relative_to(temp_root))
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            first = build(a)
            self.assertEqual(first, build(b))
            names = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
            self.assertEqual(names, sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file()))
            for name in names:
                self.assertEqual((a / name).read_bytes(), (b / name).read_bytes(), str(name))
            manifest = json.loads((a / "manifest.json").read_text(encoding="utf-8"))
            for name, entry in manifest["files"].items():
                self.assertEqual(sha((a / name).read_bytes()), entry["sha256"])
            self.assertFalse(manifest["training_readiness"]["chinese_stream_actions"])
            with self.assertRaises(FileExistsError):
                build(a)


if __name__ == "__main__":
    unittest.main()

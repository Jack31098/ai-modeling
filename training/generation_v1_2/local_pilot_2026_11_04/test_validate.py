import json
import unittest

from .validate import validate


class GeometryTests(unittest.TestCase):
    def check(self, text, phrases, protect=(), eof=True):
        return validate({"text": text, "protected_strings": list(protect), "true_document_end": eof}, json.dumps({"phrases": phrases}))

    def test_original_unicode_byte_offsets(self):
        out = self.check("e\u0301𠀀 😀", [["e\u0301", "𠀀"], [" ", "😀"]], eof=False)
        self.assertTrue(out["valid"])
        structure = out["candidate_structure"]
        self.assertEqual(structure["words"][1]["utf8"], [3, 7])
        self.assertEqual(structure["words"][-1]["utf8"], [8, 12])
        permissions = structure["supervision_permissions"]
        self.assertFalse(permissions["input_complete_is_true_document_eof"])
        self.assertFalse(permissions["allow_stream_action_forcing"])
        self.assertEqual(permissions["student_observation_fields"], ["text", "input_complete"])

    def test_normalization_and_omission_rejected(self):
        self.assertFalse(self.check("e\u0301 ", [["é", " "]])["valid"])
        self.assertFalse(self.check("a\r\nb", [["a", "\n", "b"]])["valid"])

    def test_combining_and_whitespace_boundaries(self):
        self.assertFalse(self.check("e\u0301", [["e", "\u0301"]])["valid"])
        self.assertFalse(self.check("a b", [["a b"]])["valid"])
        self.assertTrue(self.check("a\t b", [["a", "\t ", "b"]])["valid"])

    def test_protected_span(self):
        self.assertFalse(self.check("17.30元", [["17", ".", "30", "元"]], ["17.30"])["valid"])
        self.assertTrue(self.check("17.30元", [["17.30", "元"]], ["17.30"])["valid"])

    def test_no_phony_semantic_accuracy(self):
        out = self.check("研究生命起源", [["研究生", "命", "起源"]])
        self.assertTrue(out["valid"])
        self.assertIn("unreviewed", out["semantic_quality"])

    def test_budget_and_extra_labels(self):
        self.assertFalse(self.check("甲" * 65, [["甲" * 33, "甲" * 32]])["valid"])
        self.assertFalse(validate({}, '{"phrases":[["a"]],"summary":"a"}')["valid"])


if __name__ == "__main__":
    unittest.main()

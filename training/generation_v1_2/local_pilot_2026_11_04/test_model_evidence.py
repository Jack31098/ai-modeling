import unittest

from .model_evidence import count_scopes


class ScopeTests(unittest.TestCase):
    def test_whole_extra_mtp_block_not_only_named_projection(self):
        data = {"metadata": {"general.architecture":"qwen35", "qwen35.block_count":65, "qwen35.nextn_predict_layers":1},
                "stored_tensor_elements":1000,
                "tensors":[{"name":"blk.63.ffn_down.weight","elements":700},
                           {"name":"blk.64.ffn_down.weight","elements":200},
                           {"name":"blk.64.nextn.eh_proj.weight","elements":100}]}
        out = count_scopes(data)
        self.assertEqual(out["main_block_count_inferred"], 64)
        self.assertEqual(out["mtp_candidate_tensor_elements"], 300)
        self.assertEqual(out["excluding_mtp_candidate_elements"], 700)

    def test_moe_expert_count_is_not_active_parameter_count(self):
        data = {"metadata": {"general.architecture":"deepseek2", "deepseek2.block_count":47,
                             "deepseek2.expert_count":64, "deepseek2.expert_used_count":4},
                "stored_tensor_elements":1000, "tensors":[{"name":"blk.46.ffn_down_exps.weight","elements":1000}]}
        out = count_scopes(data)
        self.assertEqual(out["mtp_candidate_tensor_elements"], 0)
        self.assertEqual(out["stored_tensor_elements"], 1000)
        self.assertNotIn("active_parameters", out)


if __name__ == "__main__":
    unittest.main()

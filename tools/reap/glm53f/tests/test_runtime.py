import unittest
from types import SimpleNamespace
import numpy as np
import torch

from glm_reap.bank import Bank, Candidate
from glm_reap.runtime import StreamLinear


class GradientTests(unittest.TestCase):
    def test_small_kda_chunks_match_default(self):
        from transformers.models.glm5_next.modeling_glm5_next import chunk_kimi_delta_attention
        torch.manual_seed(7)
        q, k, v = [torch.randn(1, 97, 2, 8) for _ in range(3)]
        g = -torch.rand_like(q) * 0.1
        beta = torch.rand(1, 97, 2)
        options = dict(g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        expected, _ = chunk_kimi_delta_attention(q, k, v, chunk_size=64, **options)
        actual, _ = chunk_kimi_delta_attention(q, k, v, chunk_size=16, **options)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_export_bank_keeps_selected_bytes_and_group_mapping(self):
        from glm_reap.checkpoint import group_name
        bank = Bank()
        names = [f"model.language_model.layers.0.mlp.experts.{i}.gate_proj.weight" for i in (0, 1)]
        for i, name in enumerate(names):
            a = np.full((2, 16), i+1, np.float16)
            b = np.full((2, 16), i+2.125, np.float32)
            bank.add(name, [Candidate("F16", a.shape, a.view(np.uint8).reshape(-1)), Candidate("F32", b.shape, b.view(np.uint8).reshape(-1))])
        group = group_name(names[0])
        bank.selection[group] = 1
        before = {name: bank.read(name).copy() for name in names}
        old_bytes = bank.bytes
        bank.retain_selected()
        self.assertLess(bank.bytes, old_bytes)
        self.assertEqual(bank.original_selection[group], 1)
        self.assertEqual(bank.selection[group], 0)
        for name in names:
            np.testing.assert_array_equal(bank.read(name), before[name])
            self.assertEqual(len(bank.tensors[name]), 1)
        bank.retain_selected()
        self.assertEqual(bank.original_selection[group], 1)

    def test_task_gradient_matches_dense_mixture(self):
        rng = np.random.default_rng(42)
        a = rng.normal(size=(7, 16)).astype(np.float32)
        b = rng.normal(size=(7, 16)).astype(np.float32)
        bank = Bank()
        bank.add("test.weight", [Candidate("F32", a.shape, a.view(np.uint8).reshape(-1)), Candidate("F32", b.shape, b.view(np.uint8).reshape(-1))])
        p = torch.tensor([0., 1.], requires_grad=True)
        x = torch.tensor(rng.normal(size=(3, 16)).astype(np.float32), requires_grad=True)
        runtime = SimpleNamespace(tile=3, capture=None, probabilities={"test.weight": p}, weight=bank.read)
        output = StreamLinear(runtime, "test.weight", 7)(x)
        loss = output.square().sum()
        loss.backward()
        p_ref = p.detach().clone().requires_grad_()
        x_ref = x.detach().clone().requires_grad_()
        reference = x_ref @ (p_ref[0]*torch.tensor(a)+p_ref[1]*torch.tensor(b)).T
        reference.square().sum().backward()
        torch.testing.assert_close(output, reference)
        torch.testing.assert_close(x.grad, x_ref.grad)
        torch.testing.assert_close(p.grad, p_ref.grad)

    def test_hybrid_model_task_backward(self):
        """Exercise actual HF KDA, sparse attention, mHC and pruned MoE."""
        import copy
        import json
        from pathlib import Path
        from glm_reap.runtime import Runtime, source_name
        from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer
        model_config = Path("/mnt/nvme01/models/glm53f/base/config.json")
        if not model_config.exists():
            self.skipTest("Target GLM config is needed for the hybrid integration fixture")
        hp = copy.deepcopy(json.loads(model_config.read_text())["text_config"])
        hp.update(hidden_size=256, intermediate_size=256, moe_intermediate_size=256, num_hidden_layers=4, num_attention_heads=2, n_routed_experts=16, num_experts_per_tok=2, vocab_size=256, q_lora_rank=32, kv_lora_rank=32, index_n_heads=2, index_head_dim=16, index_topk=4, index_kpool=2, pad_token_id=None)
        hp["layer_types"] = hp["layer_types"][:4]
        hp["num_key_value_heads"] = 2
        hp["mlp_layer_types"] = ["dense", "sparse", "sparse", "sparse"]
        hp["indexer_types"] = ["full"]*4
        hp["linear_attn_config"].update(num_heads=2, kda_layers=[0, 1, 2], full_attn_layers=[3])
        cfg = Glm5NextTextConfig(**hp)
        rng = np.random.default_rng(1)
        values = {}
        for layer in range(4):
            with torch.device("meta"):
                module = Glm5NextTextDecoderLayer(cfg, layer)
            prefix = f"model.language_model.layers.{layer}."
            for name, parameter in list(module.named_parameters())+list(module.named_buffers()):
                if parameter is None:
                    continue
                shape = tuple(parameter.shape)
                value = rng.normal(0, .02, shape).astype(np.float32)
                if "norm" in name and name.endswith("weight"):
                    value.fill(1)
                if name.endswith("scale"):
                    value.fill(1)
                if name.endswith("A_log") or "e_score_correction_bias" in name:
                    value.fill(0)
                if name.endswith("dt_bias"):
                    value.fill(-2)
                if name == "mlp.experts.gate_up_proj":
                    for expert in range(16):
                        values[prefix+f"mlp.experts.{expert}.gate_proj.weight"] = value[expert, :256]
                        values[prefix+f"mlp.experts.{expert}.up_proj.weight"] = value[expert, 256:]
                elif name == "mlp.experts.down_proj":
                    for expert in range(16):
                        values[prefix+f"mlp.experts.{expert}.down_proj.weight"] = value[expert]
                elif name == "self_attn.conv1d.weight":
                    for part, chunk in zip(("q", "k", "v"), np.split(value, 3)):
                        values[prefix+f"self_attn.{part}_conv1d.weight"] = chunk
                else:
                    values[prefix+source_name(name)] = value
        values["model.language_model.norm.weight"] = np.ones(256, np.float32)
        for name in ("model.language_model.embed_tokens.weight", "lm_head.weight"):
            values[name] = rng.normal(0, .02, (256, 256)).astype(np.float32)
        class Fixture:
            config = {"text_config": hp}
            tensors = values
            def read(self, name, start=0, stop=None):
                return values[name][start:stop].copy()
        name = "model.language_model.layers.0.self_attn.f_a_proj.weight"
        bank = Bank()
        a = values[name]
        bank.add(name, [Candidate("F32", a.shape, a.view(np.uint8).reshape(-1)), Candidate("F32", a.shape, (a*1.1).view(np.uint8).reshape(-1))])
        runtime = Runtime(Fixture(), {"tile_rows": 32, "host_limit_gib": 8, "gpu_limit_gib": 1}, selected={str(i): list(range(8)) for i in (1, 2, 3)}, bank=bank)
        p = torch.tensor([0., 1.], requires_grad=True)
        runtime.probabilities[name] = p
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(4)
            with torch.no_grad():
                records = [([1, 2, 3, 4],), ([5, 6, 7],)]
                expected = [runtime.hidden(record[0]).cpu() for record in records]
                actual = runtime.hidden_batch(records)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b)
            loss = runtime.loss(list(range(1, 9)), [False]+[True]*7, checkpoint=True)
            loss.backward()
        finally:
            torch.set_num_threads(previous_threads)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(p.grad)
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(p.grad.abs().sum()), 0)

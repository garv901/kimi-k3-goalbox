"""Kimi oracle: KDA state layout, batch rows, reset semantics, weights and band grading (CPU, tiny geometry)."""

from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from ..cases import select_cases
from ..tasks import kimi
from ..tasks.quantization import unpack_mxfp4


def development(batch=3):
    case = select_cases('all', ['kimi-k3-dev-l4-tp1-b8-s8192'])[0]
    return replace(case, params=case.params | {
        'batch': batch, 'layers': 5, 'hidden': 64, 'latent_hidden': 64, 'intermediate': 64,
        'dense_intermediate': 128, 'q_heads': 4, 'head_dim': 16, 'q_lora_rank': 32,
        'kv_lora_rank': 32, 'qk_nope_dim': 8, 'qk_rope_dim': 8, 'v_head_dim': 16,
        'vocab': 128, 'experts': 8, 'topk': 2, 'context': 4, 'attn_res_block_size': 2})


class KimiReferenceTests(unittest.TestCase):
    def test_kda_v_first_matches_closed_form_transition_over_rollout(self):
        torch.manual_seed(5)
        heads, dim = 3, 16
        state = torch.randn(heads, dim, dim)
        expected = state.clone()
        A, dt = torch.randn(heads), torch.randn(heads, dim)
        for _ in range(8):
            q, k, v, g = (torch.randn(heads, dim) for _ in range(4))
            b = torch.randn(heads)
            nq = q/(q.square().sum(-1, keepdim=True)+1e-6).sqrt()/dim**0.5
            nk = k/(k.square().sum(-1, keepdim=True)+1e-6).sqrt()
            decay = torch.exp(-5*torch.sigmoid(torch.exp(A)[:, None]*(g+dt)))
            update = torch.eye(dim)[None]-b.sigmoid()[:, None, None]*nk[:, :, None]*nk[:, None, :]
            expected = torch.bmm(expected, torch.diag_embed(decay))@update + b.sigmoid()[:, None, None]*v[:, :, None]*nk[:, None, :]
            output, state = kimi.kda_step(q, k, v, g, b, A, dt, state)
            torch.testing.assert_close(state, expected, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(output, (expected@nq[:, :, None]).squeeze(-1), rtol=2e-5, atol=2e-6)

    def test_synthetic_weights_match_declared_shapes(self):
        case = development()
        values = kimi.synthetic_weights(case, 3, 'cpu', 0)
        shapes = kimi.weight_shapes(case)
        self.assertEqual(set(values), set(shapes))
        for name, (shape, dtype, _, _) in shapes.items():
            self.assertEqual((tuple(values[name].shape), values[name].dtype), (shape, dtype), name)
        decoded = unpack_mxfp4(values['l1_gate_blocks'], values['l1_gate_scales'])
        rms = decoded.square().mean().sqrt().item()
        self.assertTrue(0.3 < rms*case.params['latent_hidden']**0.5 < 3, rms)

    def test_batch_rows_equal_single_sequence_oracle(self):
        case = development(3)
        values = kimi.make_inputs(case, 21, 'cpu', rank=0)
        values['reset'] = torch.tensor([False, True, False])
        batched = kimi.reference(case, values, serial=True)
        one = replace(case, params=case.params | {'batch': 1})
        for b in range(3):
            row = {name: (value[b:b+1] if name in ('token', 'reset') or name.endswith(('_cache', '_state')) else value)
                   for name, value in values.items()}
            single = kimi.reference(one, row, serial=True)
            torch.testing.assert_close(single['logits'][0], batched['logits'][b], rtol=2e-2, atol=2e-2)
            self.assertEqual(int(single['cache_length'][0]), int(batched['cache_length'][b]))

    def test_reset_ignores_state_and_inputs_are_preserved(self):
        case = development(2)
        values = kimi.make_inputs(case, 17, 'cpu', rank=0)
        values['reset'] = torch.tensor([True, False])
        before = {name: value.clone() for name, value in values.items()}
        actual = kimi.reference(case, values, serial=True)
        self.assertTrue(all(torch.equal(value, before[name]) for name, value in values.items()))
        self.assertEqual(actual['cache_length'].tolist(), [1, case.params['context']+1])
        changed = {name: (value+10 if name.endswith(('_cache', '_state')) else value) for name, value in values.items()}
        other = kimi.reference(case, changed, serial=True)
        self.assertTrue(torch.equal(actual['logits'][0], other['logits'][0]))
        self.assertFalse(torch.equal(actual['logits'][1], other['logits'][1]))

    def test_band_accepts_oracle_and_fp32_and_rejects_noise(self):
        case = development(2)
        values = kimi.make_inputs(case, 5, 'cpu', rank=0)
        expected = kimi.reference(case, values, serial=True)
        kimi.compare(case, values, expected, {k: v.clone() for k, v in expected.items()}, serial=True)
        fp32 = kimi.reference(case, values, serial=True, fp32=True)
        as_fp32 = {k: (fp32[k].to(v.dtype) if k not in ('expert_ids', 'next_token') else v) for k, v in expected.items()}
        kimi.compare(case, values, expected, as_fp32, serial=True)
        noisy = dict(expected, logits=expected['logits']+0.2*expected['logits'].std()*torch.randn_like(expected['logits']))
        with self.assertRaises(AssertionError):
            kimi.compare(case, values, expected, noisy, serial=True)
        shuffled = dict(expected, expert_ids=expected['expert_ids'].flip(-1))  # order inside a row is free
        kimi.compare(case, values, expected, shuffled, serial=True)
        repeated = expected['expert_ids'].clone()
        repeated[1:, :, 1] = repeated[1:, :, 0]
        with self.assertRaises(AssertionError):
            kimi.compare(case, values, expected, dict(expected, expert_ids=repeated), serial=True)

    def test_forced_routes_and_shortfall(self):
        case = development(2)
        values = kimi.make_inputs(case, 9, 'cpu', rank=0)
        own = kimi.reference(case, values, serial=True, fp32=True)
        forced = kimi.reference(case, values, serial=True, fp32=True, routes=own['expert_ids'])
        self.assertTrue(torch.equal(own['logits'], forced['logits']))
        self.assertEqual(float(forced['route_shortfall'].max()), 0.0)
        worst = own['expert_ids'].clone()   # a fixed set that is not the top-k
        for layer in range(case.params['first_dense'], case.params['layers']):
            worst[layer] = torch.arange(case.params['experts']-case.params['topk'], case.params['experts'])
        bad = kimi.reference(case, values, serial=True, fp32=True, routes=worst)
        self.assertGreater(float(bad['route_shortfall'].max()), kimi.ROUTE_FLOOR)


if __name__ == '__main__':
    unittest.main()

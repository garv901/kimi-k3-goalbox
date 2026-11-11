# Verification of the Kimi K3 oracle

We compared the oracle in `megabench/tasks/kimi.py` with the official Hugging Face code of
`moonshotai/Kimi-K3` (revision `f831ab66814297da540d832a5235f8e904f29d06`).
The script is `tools/verify_kimi_vs_hf.py`. We ran it on one B200 on 2026-11-10.

## Part 1: kernels (flash-linear-attention 0.6.0)

The Hugging Face model calls these `fla` kernels. The oracle has its own PyTorch version of each.
The inputs are random. B=3, 4 heads, head dimension 128.

| Item | Relative L2 distance, oracle to `fla` |
|---|---|
| KDA output (`fused_recurrent_kda`, V-first state) | 1.6e-3 |
| KDA recurrent state (FP32) | 7.3e-8 |
| Short convolution output (`ShortConvolution`, SiLU) | 1.7e-3 |
| Short convolution state | equal |
| Gated output norm (`FusedRMSNormGated`, sigmoid) | 1.6e-3 |

The output distances of approximately 1.6e-3 are the BF16 rounding of the kernel outputs.

## Part 2: model (4 real layers)

The script loads layers 0 to 3 of the real checkpoint (dense layer 0, KDA layers 1 and 2, MLA layer 3)
through the converted-weight loader.
It runs two decode steps for 4 sequences. Step 1 starts all sequences new.
Step 2 uses the caches and states that step 1 wrote.
The same tokens go through `KimiLinearForCausalLM` (transformers 4.56.2) with 4 layers.

| Step | Logits relative L2, oracle to HF | Logits relative L2, FP32 oracle to HF | Same top-1 token |
|---|---|---|---|
| 1 (new sequences) | 5.9e-3 | 6.2e-3 | 4 of 4 |
| 2 (state from step 1) | 8.1e-3 | not measured | 4 of 4 |

The two distances are at the level of BF16 noise.
The FP32 oracle is not nearer to HF than the BF16 oracle. Thus the HF model also has BF16 rounding.

## Details of the checkpoint that the oracle obeys

These details are not clear from the model card. We found them in the checkpoint and the code:

- `A_log` has 128 values, but the model uses only the first 96 (one for each KDA head). The remaining values are zero.
- `conv1d` weights, `o_norm` weights, `A_log` and `dt_bias` are FP32. The oracle keeps them in FP32.
- The block attention residual (`_apply_attn_res`) runs fully in FP32.
- The `q_a` and `kv_a` layer norms use eps 1e-6. All other RMS norms use eps 1e-5.
- MLA has no rotary transform. The score scale is 192^-0.5.
- The routed expert weights are MXFP4 (E2M1 codes, E8M0 scales for each block of 32).

## Harness checks (synthetic weights, B200, 2026-11-11)

Expert routing is discrete. The first version of the grader compared every output with one FP32 run that selected its own routes.
With random routers, the top-16 scores have small gaps. BF16 noise then changed approximately one expert in some rows, and that changed the row by tens of percent.
The oracle distance on the logits was 0.62 (dev-l4) and 0.96 (dev-l8-tp2). This made the band too wide.

The grader now runs the FP32 reference on fixed routes: once on the oracle routes and once on the candidate routes.
A candidate route must be a near-top-k set under the FP32 router scores.

| Case | Submission | Oracle logits distance to FP32 | Result |
|---|---|---|---|
| `kimi-k3-dev-l4-tp1-b8-s8192` | The oracle | 0.027 (before the change: 0.62) | pass |
| `kimi-k3-dev-l4-tp1-b8-s8192` | The oracle + noise of 0.02 x std on the logits | 0.027 | pass (ratio 0.76 of the band) |
| `kimi-k3-dev-l4-tp1-b8-s8192` | The oracle + noise of 0.1 x std on the logits (first grader) | 0.66 | fail (ratio 3.0) |
| `kimi-k3-dev-l8-tp2-b8-s8192` | The oracle, 2 ranks | 0.037 (before the change: 0.96) | pass |

With the change, no row has a `next_token` that is different from the FP32 argmax. The unit tests include a candidate with a repeated expert and a candidate with routes that are not near the top-k. The grader refuses both.

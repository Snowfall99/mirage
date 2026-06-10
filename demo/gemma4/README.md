# Gemma 4 on MPK (Blackwell: SM100 and SM120)

First-pass integration of `google/gemma-4-12B` (the "unified" encoder-free
variant, text path only). Written and reviewed offline — **not yet compiled or
run on a GPU**; see "Verification status" below.

## Usage

```bash
# MPK megakernel path (B200/SM100, or RTX Pro 6000 / consumer Blackwell SM120)
python demo.py --use-mirage

# PyTorch reference (requires transformers >= the version with gemma4 support)
python demo.py
```

## SM120 (RTX Pro 6000) support

SM120 is consumer/workstation Blackwell: ~99KB shared memory per block (vs
~227KB on B200), no tcgen05/TMEM, no WGMMA. MPK previously had no SM120 path
at all (`linear_layer` asserted "Unsupported compute capability"). What makes
it work now:

- `runtime_header.h`: a `MPK_TARGET_CC >= 120` rung in the shared-memory
  ladder (99KB) — previously SM120 matched the `>= 90` branch and would have
  requested 207KB.
- SM120 builds use the **ampere task set** (`-arch=native`, 128-thread
  workers, CUTLASS sm_80 GEMMs) plus the portable Blackwell tasks Gemma needs:
  the extended paged attention and `elementwise_add_sm100`, included from
  `tasks/ampere/task_header.cuh` under `MPK_TARGET_CC >= 120`. The attention
  kernel uses only sm_80-level primitives (cp.async, ldsm, mma.m16n16k16,
  named barriers), so it compiles for SM120 directly.
- Python dispatch routes cc 120 to ampere kernels for linear / rmsnorm
  (previously cc>=90 incorrectly picked `rmsnorm_hopper`), and
  `hopper/utils.cuh:wg_sync` no longer compiles to a `brkpt` trap on SM120.
- Tighter smem budget: the demo picks `kv_tile_size` 16 (sliding) / 8
  (global) and clamps `max_num_batched_tokens` to 2 on SM120.
- The lm_head/argmax split uses a fixed grid of 128 tasks (divides vocab
  262144) instead of `num_workers`, which is 144 on RTX Pro 6000 and does
  not divide the vocab.

Requires CUDA 12.8+ (for sm_120 in `-arch=native`).

## What was added for Gemma 4

| Piece | Where |
|---|---|
| `gelu_mul` task (GeGLU) | `tasks/{ampere,hopper}/gelu_mul*.cuh`, `TASK_GELU_MUL`=122 / `TASK_GELU_MUL_HOPPER`=165, `gelu_mul_layer()` |
| Sliding-window + Gemma attention | `tasks/blackwell/attention_sm100.cuh` — new template params `SLIDING_WINDOW`, `K_EQ_V`, `V_NORM`, `UNIT_SM_SCALE`, `KV_TILE_SIZE_IN` (defaults preserve existing behavior) |
| Pairwise norm+RoPE helper | `tasks/blackwell/norm_sm100.cuh` — `rms_norm_rope_pairwise_sm100`, correct for head_dim > NUM_THREADS; `weight_ptr == nullptr` = unweighted norm |
| Registration | `register_gemma4_paged_attention_sm100_task` in `task_register.cc`, dispatch name `gemma4_paged_attention_sm100` |
| Python layer | `PersistentKernel.gemma4_paged_attention_layer(...)` |
| Tests for gelu_mul | `tests/runtime_python/blackwell/sm100_gelu_mul/` |

## How the architecture maps onto MPK

- **Sliding layers** (head_dim 256, 16 Q / 8 KV heads, window 1024): one task
  per KV head, `sliding_window=1024`, `kv_tile_size=32`. The kernel skips KV
  tiles older than the window and masks the boundary tile.
- **Global layers** (head_dim 512, 16 Q heads, 1 shared KV head, K=V): the
  single K projection is replicated 4x in the fused QKV weight so the tensor
  partition gives each of 4 tasks a self-contained `[4 Q heads | K]` chunk
  (`q_split=4`, `k_eq_v=True`, `kv_tile_size=16`). All 4 tasks write identical
  bytes to the shared KV cache (benign). K gets k_norm+RoPE, V gets the
  unweighted v_norm — so the K and V caches differ even though K=V at the
  projection.
- **Proportional / partial RoPE** (global layers, factor 0.25, theta 1e6):
  zeroed inverse frequencies → cos=1/sin=0 table entries → the standard
  rotate_half kernel applies identity on the unrotated dims. Table generation
  in `make_rope_tables()` mirrors transformers'
  `_compute_proportional_rope_parameters`.
- **Softmax scale 1.0** (`UNIT_SM_SCALE`), matching `Gemma4TextAttention.scaling`.
- **Sandwich norms**: `rmsnorm_layer` -> linear -> `rmsnorm_layer` ->
  `elementwise_add_layer` (no fused-residual linear, since the residual is
  added after the post-norm). Gemma4RMSNorm is a plain `*weight` RMS norm
  (no Gemma2/3-style `1+w`), so weights load unmodified.
- **Embedding scaling**: embedding table is pre-multiplied by
  `bf16(sqrt(hidden_size))` at load; the tied lm_head uses the unscaled table.

## Known limitations

- Blackwell only (the task asserts `target_cc in (100, 120)`); no Hopper or
  Ampere attention path yet.
- `max_num_batched_tokens <= 4` on SM100 and `<= 2` on SM120: the
  head_dim-512 global task's shared-memory layout (Q/O staging +
  64-float-per-thread reduction buffer) caps per-task query rows.
- Global-attention tasks hold ~256 accumulator floats per thread
  (`o[1][32][8]`) — expect register spills; correctness-first, optimize later.
- `num_kv_shared_layers > 0` (trailing layers reusing earlier KV states, used
  by E2B/E4B) unsupported. Per-layer embeddings (E-variants) unsupported.
  The 26B-A4B MoE variant is not wired.
- Single GPU, offline mode, greedy decoding.

## Verification status / next steps on a GPU box

1. `cd tests/runtime_python/blackwell/sm100_gelu_mul && python setup.py build_ext --inplace && python test_gelu_mul.py` — validates the GeGLU kernel against `F.gelu(approximate="tanh")`.
2. `python test_gelu_mul_testmode.py` — full pipeline for the gelu_mul layer.
3. Compile check of the extended attention template at head_dim 256/512
   (smem static_asserts will fire if the budget math is off).
4. `python demo.py --use-mirage --max-new-tokens 32 --save-tokens` vs
   `python demo.py --max-new-tokens 32 --save-tokens` and diff the token ids.
5. The HF norm-weight convention was verified against transformers source
   (`Gemma3nRMSNorm`: plain `*weight`), but step 4 is the real gate.

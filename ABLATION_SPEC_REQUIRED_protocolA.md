# HybridSBD — Ablation Study Specification v2 — REQUIRED version, Protocol A only (13 runs) — R1.1 / R2.5

This is the minimum set that fully answers Reviewer 1. It answers both of R1's questions:
(a) whether **dual-label supervision itself** produces the gains rather than architectural
complexity, and (b) **which existing components matter**.

Every experiment switches off or replaces an *existing* code path. No new module is added.

**Total: 13 training runs** (5 supervision + 8 components). All other analyses below are
evaluation only and reuse the trained checkpoints.

---

## 0. Pre-run checklist (do all of this before any training)

1. **Verify every flag against the forward pass** in `detection/hybrid_boundary_detector.py`.
   Switching a flag off must remove *all* computation of that component and nothing else.
   Watch the dependencies:
   - FrameSimilarity and the temporal delta stream both read `f_pf` (Branch C).
   - The color histogram stream reads raw pixels, not branch features.
   - The edge head changes every branch embedding from 640 to 512 dimensions.
2. **Unit test each variant** with one forward pass on a dummy batch: print output shapes,
   fusion input size, and the exact integer parameter count.
3. **Fix the sampling seed.** Replace `hash(vid_name)` with a stable hash:
   ```python
   import hashlib
   stable_seed = int(hashlib.md5(vid_name.encode()).hexdigest(), 16) % (2**31)
   ```
   Also set `PYTHONHASHSEED=0` and seed `random`, `numpy`, `torch`, `torch.cuda`, and every
   DataLoader worker (`worker_init_fn`). After this change, A0 must be retrained (step 1).
4. **Check ℓ_min:** confirm where `test_clipshots.py` applies the minimum shot length and
   document it.
5. **Sliding-window details:** document how overlapping predictions are averaged, how the
   first/last frames are handled, and how an incomplete final window is padded. Identical
   for all runs.
6. **Data-overlap check:** no video in more than one of SHOT train / SHOT val / SHOT test /
   ClipShots train / ClipShots test / BBC.
7. **Benchmark hygiene:** `torch.cuda.reset_peak_memory_stats()` and
   `torch.cuda.empty_cache()` before each latency/VRAM measurement.
8. **Record the environment** per run: Python, PyTorch, CUDA versions, GPU model, git commit.

---

## 1. Reproduce the primary model (A0) — two separate checks

The code has changed over six months. Retrain the primary model from the current code first.

**Check 1 — architecture:** 4 fusion blocks; no positional encoding; delta projection
weight of shape (64, 640); D = 384; exact integer parameter count (rounds to 10.9M).

**Check 2 — performance (Protocol A, comparable to the paper):**

| Test set | Paper | Acceptable range |
|---|---|---|
| SHOT F1 | 0.924 | ≈ 0.915 – 0.930 |
| ClipShots F1 | 0.793 | ≈ 0.785 – 0.800 |
| BBC F1 | 0.971 | ≈ 0.965 – 0.975 |

- Both pass → this run is **A0**, the reference for all ablations.
- Check 1 fails → the code differs from the paper; stop and resolve first.
- Only check 2 fails → likely a pipeline difference (e.g. the new sampling seed); report to
  the team. Ablations can proceed if all use the same new code, but the main paper numbers
  would need discussion.

All runs must use the same code version and git commit as A0.

---

## 2. Evaluation protocol (Protocol A, as in the paper)

**Protocol A — per-test-set threshold selection.** Grid search on each test set:
τ_c ∈ {0.10, 0.15, …, 0.90}; τ_g ∈ {0.30, 0.40, 0.50, 0.60}; ℓ_min ∈ {1, 2, 3, 4} for
ClipShots only. Used for **all** results: the A0 reproduction check and every ablation
table. Label: `A_test_selected`.

Why this is acceptable: every variant is evaluated at its own best operating point, exactly
as in the paper's main tables. The bias from selecting thresholds on test annotations
inflates absolute F1 for every variant in the same way, so the differences between variants
remain comparable. The disclosure in Section 5.3 of the paper covers the ablations as well.

**Common:** `test_autoshot.py`, `test_clipshots.py`, `test_bbc.py`; sliding window T = 128,
stride 64, averaging; `predictions_to_scenes` + `evaluate_scenes(n_frames_miss_tolerance=2)`.
Single-output variants (A1, A3, A2) use only the τ_c grid.

**Efficiency:** `benchmark_model.py`, B = 16, FP32, 10 warm-up + 80 timed iterations,
synthetic input, peak memory reset first. Report as **model inference latency** (forward
pass only).

---

## 3. Common training protocol

Only the ablated item changes.

- **Model:** D = 384, L = 4, T = 128, input 48×64, dropout rates as in the paper.
- **Data:** SHOT 648 training videos, video-level 15% validation split (`random_seed = 42`,
  same validation videos for every run); ClipShots `clipshots_mix_fraction = 0.7`, redrawn
  every epoch.
- **Augmentation:** flip p = 0.5; brightness ±0.2, contrast ±0.15, noise σ = 0.02 (each
  p = 0.5); clip-level brightness ±0.1, p = 0.5.
- **Optimization:** AdamW, lr 5e-5, weight decay 0.05, batch 32, up to 80 epochs, 5-epoch
  warm-up, cosine to 5e-6, AMP, gradient clipping 1.0.
- **Early stopping:** patience 15 on SHOT validation F1, using that variant's inference rule.
- **Dual-label loss:** λ1 = 5.0, λ2 = 0.3, α_y = 0.80, α_z = 0.50, γ = 2.0, label
  smoothing 0.1 on z.
- **Seed:** 42 for all runs.

---

## 4. Code changes: configuration flags

Defaults reproduce the primary model exactly.

```python
ablation = {
    "supervision":   "dual",   # "dual" | "y_only" | "unified"
    "use_branch_b":  True,     # difference-based feature branch
    "use_branch_c":  True,     # per-frame pathway (f_pf)
    "rel_source":    "pf",     # input of FrameSimilarity + delta: "pf" | "rgb"
    "use_fsim":      True,     # FrameSimilarity
    "use_csim":      True,     # ColorHistogramSimilarity
    "use_delta":     True,     # temporal delta stream
    "use_motion":    True,     # pixel-level motion descriptor m(t)
    "use_edge_head": True,     # edge-density head
    "use_attention": True,     # MHA + gate in Progressive Fusion Blocks
}
```

**Fusion input size (computed, never hard-coded), D = 384:**

| Stream | Proj. dim | Flag |
|---|---|---|
| RGB spatial (Branch A) | 192 | always on |
| Difference branch (Branch B) | 192 | `use_branch_b` |
| FrameSimilarity | 96 | `use_fsim` |
| ColorHistSim | 96 | `use_csim` |
| Temporal delta | 64 | `use_delta` |
| Pixel motion m | 15 | `use_motion` |
| **Full total** | **655** | |

```python
fusion_in = D//2
fusion_in += D//2 if cfg.use_branch_b else 0
fusion_in += D//4 if cfg.use_fsim     else 0
fusion_in += D//4 if cfg.use_csim     else 0
fusion_in += 64   if cfg.use_delta    else 0
fusion_in += 15   if cfg.use_motion   else 0
```

- **Edge head off:** `emb_dim = 640 if cfg.use_edge_head else 512`, used everywhere a branch
  embedding is consumed. No layer hard-coded to 640.
- **Attention off:** skip MHA *and* gate: `h_blend = h_local`.
- **Branch C off:** do not compute `f_pf`; FrameSimilarity and delta use `f_rgb`
  (`rel_source = "rgb"`).
- **Single head (A2):** instantiate only one prediction head.

---

## 5. The 13 required runs

### Supervision (5 runs)

| # | ID | Architecture | Heads | Supervision | Training loss | Inference rule |
|---|---|---|---|---|---|---|
| 1 | **A0** | Full | 2 | Dual-label (primary) | λ1·Focal(y) + λ2·Focal(z̃) | b = (p_z > τ_g) ∨ (p_y > τ_c) |
| 2 | **A1** | Full | 2 | Midpoint only | λ1·Focal(y); λ2 = 0 | b = p_y > τ_c |
| 3 | **A3** | Reduced | 2 | Midpoint only | as A1 | as A1 |
| 4 | **A4** | Reduced | 2 | Dual-label | as A0 | as A0 |
| 5 | **A2** | Full | 1 | Unified, u = y ∨ z | Focal(u), λ = 1, γ = 2, α_u = 0.8 | b = p_u > τ |

- **A1:** z-head kept but receives no gradient and is ignored at inference, so parameters
  equal A0.
- **A2:** one head only; report its parameter count (≈ 0.2M below A0). No label smoothing.
- **A3/A4:** `use_branch_b=False, use_branch_c=False, use_fsim=False, use_csim=False,
  use_delta=False, use_motion=False` (edge head stays on). Fusion input = 192.

### Components (8 runs, leave-one-out from A0, all dual-label)

| # | ID | Description (use this wording in the paper) | Flags changed |
|---|---|---|---|
| 6 | **C2a** | Without FrameSimilarity | `use_fsim=False` |
| 7 | **C2b** | Without ColorHistogramSimilarity | `use_csim=False` |
| 8 | **C4** | Without the per-frame pathway; relational features computed from Branch A | `use_branch_c=False, rel_source="rgb"` |
| 9 | **C6** | Without self-attention and the attention gate | `use_attention=False` |
| 10 | **C1** | Without the difference-based feature branch (Branch B) | `use_branch_b=False` |
| 11 | **C3** | Without the temporal delta stream | `use_delta=False` |
| 12 | **C5a** | Without the pixel-level motion descriptor | `use_motion=False` |
| 13 | **C5b** | Without the edge-density head | `use_edge_head=False` |

Runs are listed in recommended order: the supervision runs first, then the components most
likely to matter.

---

## 6. Evaluation-only analyses (recommended — no training, reuse checkpoints)

### 6.1 Per-type analysis on ClipShots (A0, A1, A2)

GT type: length ≤ 1 frame = cut, > 1 = gradual. Use each model's ClipShots thresholds (Protocol A).

| Metric | Definition | A0 | A1 | A2 |
|---|---|---|---|---|
| Cut recall | Fraction of GT cuts matched (type-agnostic matching) | ✓ | ✓ | ✓ |
| Gradual recall | Fraction of GT gradual transitions matched | ✓ | ✓ | ✓ |
| Gradual F1 | Predicted gradual = transitions with ≥ 1 frame p_z > τ_g, matched against GT gradual only | ✓ | N/A | N/A |
| Gradual IoU | Temporal IoU of matched GT gradual vs. predicted interval (contiguous frames above threshold, not scene intervals) | ✓ | N/A | ✓ |

### 6.2 Duration groups (A0 vs C6)

Recall on GT gradual transitions grouped by length: ≤ 10, 11–30, > 30 frames (report n per
group). Hypothesis: C6 drops mostly in the > 30 group.

---

## 7. What to record (one CSV row per run)

```
run_id, seed, git_commit, python_version, pytorch_version, cuda_version, gpu_model,
supervision, flags_json, params_total, params_trainable, best_epoch, train_hours,
evaluation_protocol, val_F1,
shot_tau_c, shot_tau_g, shot_P, shot_R, shot_F1, shot_TP, shot_FP, shot_FN,
cs_tau_c, cs_tau_g, cs_lmin, cs_P, cs_R, cs_F1, cs_TP, cs_FP, cs_FN,
bbc_tau_c, bbc_tau_g, bbc_P, bbc_R, bbc_F1, bbc_TP, bbc_FP, bbc_FN,
latency_B16_ms, vram_B16_MB
```
Save every checkpoint and the threshold-search JSON.

---

## 8. Result tables to fill (Protocol A, as in the main results)

**Table 7a — Supervision**

| ID | Variant | Heads | Params | SHOT F1 | CS F1 | BBC F1 | Latency (ms) |
|---|---|---|---|---|---|---|---|
| A0 | Full, dual-label (primary) | 2 | | | | | |
| A1 | Full, midpoint only | 2 | | | | | |
| A2 | Full, unified single head | 1 | | | | | |
| A3 | Reduced, midpoint only | 2 | | | | | |
| A4 | Reduced, dual-label | 2 | | | | | |

**Table 7b — Components (dual-label)**

| ID | Description | Params | SHOT F1 | CS F1 | BBC F1 | Latency (ms) |
|---|---|---|---|---|---|---|
| A0 | Full model | | | | | |
| C1 | w/o difference-based branch (B) | | | | | |
| C2a | w/o FrameSimilarity | | | | | |
| C2b | w/o ColorHistogramSimilarity | | | | | |
| C3 | w/o temporal delta | | | | | |
| C4 | w/o per-frame pathway (relations from Branch A) | | | | | |
| C5a | w/o pixel-level motion descriptor | | | | | |
| C5b | w/o edge-density head | | | | | |
| C6 | w/o self-attention and gate | | | | | |

**Table 8 — Per-type analysis on ClipShots** (if 6.1 is done)

| Model | Cut recall | Gradual recall | Gradual F1 | Gradual IoU |
|---|---|---|---|---|
| A0 | | | | |
| A1 | | | N/A | N/A |
| A2 | | | N/A | |

**Table 9 — Recall by gradual-transition duration** (if 6.2 is done)

| Model | ≤ 10 (n = ) | 11–30 (n = ) | > 30 (n = ) |
|---|---|---|---|
| A0 | | | |
| C6 | | | |

---

## 9. Interpreting the results (single seed)

Hypotheses only; report the numbers as they come out.

**Supervision** (ΔF1_full = A0 − A1, ΔF1_reduced = A4 − A3):

| Case | Pattern | What the paper can claim |
|---|---|---|
| 1 | Both clearly positive (≥ 0.01) | Dual-label supervision helps independently of architectural complexity |
| 2 | Positive only for the full architecture | The benefit depends on the additional components; say so |
| 3 | Neither ≥ 0.01 | No strong accuracy claim; reframe: the extent head adds interval prediction and cut/gradual typing (Table 8) at no accuracy cost |

| Comparison | Hypothesis | If not confirmed |
|---|---|---|
| A0 vs A2 | A0 higher | Remove wording that a unified target is harmful (Section 2.4) |
| C2a / C2b | Largest drops | State which stream carries the boundary signal |
| C4 | Lower cut recall than A0 | Weaken the Branch C argument (Sections 4.2.2, 4.3.1) |
| C6 | Lower recall on > 30-frame transitions | Note convolutions suffice; faster variant |
| Any Cx | Drop | No drop: state the component is not needed — still a valid answer to R1 |

**Single-seed caveat:** with one seed per variant, treat differences below 0.01 F1 as
inconclusive, and state in the paper that each variant was trained once. If the A0 − A1
difference turns out small (< 0.01), the 4 seed runs from the FULL version (A0 and A1 with
seeds 1337 and 2024) are the first thing to add if time allows.

---

## 10. What to send back for the paper update

1. The CSV (Section 7), all 13 runs.
2. Tables 8 and 9 numbers, if done.
3. Notes on any odd run and the outcome of the A0 reproduction checks.

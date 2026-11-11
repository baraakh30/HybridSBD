# Results

Summary of the HybridSBD ablation study. Models are trained on SHOT (448 training videos; the 200 SHOT test videos are excluded) and ClipShots, and evaluated on the SHOT test set (200 videos), the ClipShots test set (500 videos) and BBC Planet Earth (11 episodes) with scene-based F1 (±2 frames), ℓ_min = 1 (no minimum-shot-length merging) and the sliding-window protocol of the AutoShot benchmark. Every table shows two threshold rules: **Protocol A** selects τ_c and τ_g on each test set (optimistic, comparable only to other Protocol-A numbers); **Protocol B** selects them once on the 67 SHOT validation videos and applies them unchanged. The full tables, methodology, training details, speed measurements and caveats are in [`reports/RESULTS_FULL.md`](../reports/RESULTS_FULL.md); differences below 0.01 F1 are treated as inconclusive.

## Headline: full model (A0) and its midpoint-only counterpart (A1)

F1, mean ± sample SD over seeds 42, 43 and 44.

| Model | Protocol | SHOT | ClipShots | BBC |
|---|---|---|---|---|
| A0 (dual-label, full) | A | 0.8466 ± 0.0018 | 0.7990 ± 0.0047 | 0.9693 ± 0.0034 |
| A0 (dual-label, full) | B | 0.8268 ± 0.0202 | 0.7633 ± 0.0039 | 0.9446 ± 0.0066 |
| A1 (midpoint only, full) | A | 0.8458 ± 0.0021 | 0.7569 ± 0.0043 | 0.9686 ± 0.0015 |
| A1 (midpoint only, full) | B | 0.8281 ± 0.0046 | 0.7374 ± 0.0032 | 0.9579 ± 0.0029 |

Paired A0 − A1 on ClipShots: Protocol A +0.0392, +0.0366, +0.0507; Protocol B +0.0305, +0.0228, +0.0242 (seeds 42, 43, 44). Dual-label supervision helps on ClipShots at every seed under both protocols. On SHOT and BBC the paired A0 − A1 differences (seeds 42, 43, 44) are: SHOT Protocol A: -0.0029, +0.0003, +0.0049 (sign changes across seeds); SHOT Protocol B: -0.0230, -0.0067, +0.0255 (sign changes across seeds); BBC Protocol A: +0.0021, -0.0014, +0.0015 (sign changes across seeds); BBC Protocol B: -0.0158, -0.0162, -0.0080 (same sign at all seeds).

## Supervision (seed 42), F1 as Protocol A / Protocol B

| Run | Description | SHOT | ClipShots | BBC |
|---|---|---|---|---|
| A0 | Full, dual-label (primary) | 0.8449 / 0.8084 | 0.8001 / 0.7658 | 0.9719 / 0.9439 |
| A1 | Full, midpoint only | 0.8478 / 0.8313 | 0.7609 / 0.7352 | 0.9698 / 0.9597 |
| A2 | Full, unified single head | 0.8509 / 0.8350 | 0.7709 / 0.7589 | 0.9473 / 0.9473 |
| A3 | Reduced, midpoint only | 0.8312 / 0.8223 | 0.7606 / 0.7562 | 0.9714 / 0.9713 |
| A4 | Reduced, dual-label | 0.8370 / 0.8364 | 0.8045 / 0.7627 | 0.9721 / 0.9660 |

## Build-up ladder (seed 42), F1 as Protocol A / Protocol B

| Step | Run | SHOT | ClipShots | BBC |
|---|---|---|---|---|
| 0 | M0_min | 0.8169 / 0.8124 | 0.7555 / 0.7396 | 0.9543 / 0.9502 |
| 1 | M1_edge | 0.8261 / 0.8082 | 0.7672 / 0.7633 | 0.9584 / 0.9513 |
| 2 | B_noatt | 0.8351 / 0.8259 | 0.7984 / 0.7791 | 0.9682 / 0.9525 |
| 3 | A4 | 0.8370 / 0.8364 | 0.8045 / 0.7627 | 0.9721 / 0.9660 |
| 4 | S1_rel | 0.8392 / 0.8275 | 0.8054 / 0.7862 | 0.9706 / 0.9534 |
| 5 | A0 | 0.8449 / 0.8084 | 0.8001 / 0.7658 | 0.9719 / 0.9439 |

## Leave-one-out components (seed 42), change in F1 relative to A0 as Protocol A / Protocol B

| Run | Removed | SHOT | ClipShots | BBC |
|---|---|---|---|---|
| C1 | difference-based branch (B) | +0.0087 / +0.0274 | -0.0059 / +0.0101 | -0.0017 / +0.0005 |
| C2a | FrameSimilarity | +0.0008 / +0.0325 | +0.0008 / -0.0393 | -0.0037 / +0.0009 |
| C2b | ColorHistogramSimilarity | -0.0089 / +0.0176 | +0.0076 / -0.0144 | +0.0012 / +0.0107 |
| C3 | temporal delta | -0.0086 / +0.0124 | -0.0045 / +0.0052 | +0.0009 / +0.0144 |
| C4 | per-frame pathway (relations from Branch A) | -0.0029 / +0.0226 | +0.0067 / +0.0262 | -0.0020 / +0.0165 |
| C5a | pixel-level motion descriptor | +0.0037 / +0.0070 | +0.0020 / +0.0294 | -0.0003 / +0.0224 |
| C5b | edge-density head | +0.0004 / +0.0257 | -0.0034 / -0.0020 | +0.0015 / +0.0241 |
| C6 | self-attention and gate | -0.0085 / +0.0063 | -0.0032 / +0.0050 | -0.0052 / +0.0095 |

Largest single-component change on any test set: 0.0089 (Protocol A), 0.0393 (Protocol B).

## Where to find what

- [`reports/RESULTS_FULL.md`](../reports/RESULTS_FULL.md): all tables (precision / recall / thresholds for every run, per-type analysis, duration analysis, seeds, interpretation, speed).
- [`reports/results_all_runs.csv`](../reports/results_all_runs.csv): one row per run, seed, protocol and test set.
- [`docs/IMPLEMENTATION_NOTES.md`](IMPLEMENTATION_NOTES.md): implementation details of the evaluation and training code.

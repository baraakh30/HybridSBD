- **Code version:** commit `b632924` of the project repository (training code unchanged since `aee8e0e`, which added the `--split_seed` option; `b632924` adds the Protocol B and re-evaluation scripts).
- **Training commands (4 runs, in parallel on one GPU):**
  `bash run_seeds.sh` which, for each of A0:43 A0:44 A1:43 A1:44, runs
  `OUT=<outputs>/seed_runs/<RUN>_s<SEED> RUN_ID=<RUN> bash run_experiment.sh <RUN>_s<SEED> --synth_prob 0.25 --ema_decay 0.999 --strong_aug --seed <SEED> --split_seed 42`
- **Protocol B, validation step:** `python predecode_shot_val.py ...` (once), then `RUNS="<run ...>" bash protocolB_val.sh` (runs `test_autoshot.py` on the 67 validation videos with `--min_scene_lengths 1`).
- **Evaluation:** `python reeval_lmin1.py --runs <21 runs>` (Protocol A, ℓ_min = 1) and `python protocolB_eval.py --runs <21 runs>` (Protocol B); `python make_combined_csv.py` writes `results_all_runs.csv`; `python make_followup_report.py` writes this report.
- **Seed-42 regression:** after adding the four new runs, all 17 seed-42 results of both evaluations are identical to the earlier run (only timing fields differ).

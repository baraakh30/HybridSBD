# Commands for the follow-up experiments

Run from the repository root. Paths below follow the training-server layout (`/workspace/data`, `/workspace/outputs`).

1. **Extra seeds of A0 and A1** (seeds 43 and 44, one GPU, four jobs at once; the train / validation split stays at `--split_seed 42`):

   ```bash
   bash scripts/run_queue.sh seeds
   ```

   This runs `scripts/run_experiment.sh <RUN>_s<SEED> --synth_prob 0.25 --ema_decay 0.999 --strong_aug --seed <SEED> --split_seed 42`
   for each of A0:43, A0:44, A1:43 and A1:44, with `OUT=<outputs>/seed_runs/<RUN>_s<SEED>` and `RUN_ID=<RUN>`.

2. **Protocol B, validation step** (decode the 67 SHOT validation videos as full videos once, then evaluate each model on them):

   ```bash
   python data_prep/predecode_shot_val.py ...
   RUNS="A0 A1 A2 ..." bash scripts/rerun_evaluations.sh validation
   ```

3. **Re-evaluation from the saved probability tracks** (no model is run):

   ```bash
   python evaluation/reevaluate.py lmin1 --data_root /workspace/data --outputs /workspace/outputs --runs <runs>
   python evaluation/reevaluate.py protocol-b --data_root /workspace/data --outputs /workspace/outputs --runs <runs>
   ```

   Optional check that the saved probabilities equal a fresh forward pass:

   ```bash
   RUNS="A0 A1" bash scripts/rerun_evaluations.sh fresh
   python checks/verify.py fresh-pass A0 A1 --saved /workspace/outputs/ablations_v2 --fresh /workspace/outputs/reeval_lmin1/fresh --output fresh_vs_saved.json
   ```

4. **Reports** (from the results copied to `server_results/`, see `scripts/sync.sh`):

   ```bash
   python report_tools/build_reports.py all --pdf
   ```

   writes `reports/FOLLOWUP_RESULTS.md`, `reports/RESULTS_FULL.md`, `docs/RESULTS.md`, `reports/results_all_runs.csv` and the two PDFs.

# TODO

Scope: **E1 (`task_id` baseline) vs E2 (`supervised` context encoder)** on MT30.
Encoder-family work (`pearl`, `varibad`, the K/d_c/α/CE-vs-NCE ablation sweep)
is parked — see "Parked" at the bottom, not deleted, just not on the critical
path right now.

## 0. Ship what's already written (blocks everything below)

- [ ] Commit + push the uncommitted local changes, then pull/rebuild on the
      cluster. Nothing below takes effect until this happens.
  - `tdmpc2/tdmpc2.py` — refactored `act()` to expose a pure `context()`
    accessor (behavior-preserving; fixed a `@torch.no_grad()` decorator bug
    introduced by the refactor itself).
  - `tdmpc2/evaluate_carl.py` — logs `consistency_error` / `reward_error`
    per scenario (held-out model-prediction error, RQ1/RQ3); new `--probe`
    mode (see #3).
  - `tdmpc2/config.yaml` + `tdmpc2/common/logger.py` — new `wandb_name_prefix`
    key (default "" = no behavior change) prepended to the wandb run name.
  - `slurm/hpc/eval_carl_sweep_taskid_supervised.sh` — combos alternated
    supervised/task_id per seed; `RERUN_TAG` support, **defaulting to
    `secondrun`** (see #1 — this changes what a bare resubmit does).
  - `slurm/hpc/eval_carl_probe_taskid_supervised.sh` — new (see #3).
  - `slurm/hpc/run_ctxsweep_supervised_seed5_ampere.sh` — new, parked (see
    below).
- [ ] Delete the two stray untracked patch files at repo root — content is
      already merged, they're just clutter:
      `0001DeriveInfoNCEviewsbysplittingthecontextwindow.patch`,
      `0002varibadelbofixrebased(1).patch`.

## 1. Re-run CARL eval to backfill the new diagnostic columns

The task_id/supervised CARL sweep (`slurm/hpc/eval_carl_sweep_taskid_supervised.sh`)
already ran, but with the *old* `evaluate_carl.py` — no `consistency_error`/
`reward_error` in those results. Return/success/retention numbers from that
run are still valid (the refactor is behavior-preserving, no new randomness
for `task_id`/`supervised`) — don't discard them, just backfill.

- [ ] Resubmit `sbatch slurm/hpc/eval_carl_sweep_taskid_supervised.sh`.
      The script now defaults to `RERUN_TAG=secondrun`: it writes to fresh
      work_dirs (`logs/mt30/<seed>/secondrun_carl_...`) and fresh wandb
      projects (`secondrun_taskid_generalizable` /
      `secondrun_supervised_generalizable`), so **no `rm` of resume markers or
      CSVs is needed** and the original run's data is untouched. (To backfill
      in place under the original project names instead, pass
      `--export=ALL,RERUN_TAG=` — only then does the old clear-both-first
      dance apply.)
- [ ] Fast backfill is now the DEFAULT (~6-7x cheaper): `EVAL_EPISODES=3`
      and `CKPT_STRIDE=2` (every other checkpoint; `final.pt`, the fully-
      trained model, is ALWAYS kept — it is the only copy of the 3M model,
      not a duplicate of any numeric ckpt). The error columns are per-step
      averages so they converge with few episodes; downstream therefore
      merges: **returns/retention from run 1, error columns from
      `secondrun_*`**. For a full-budget replacement run instead:
      `sbatch --export=ALL,EVAL_EPISODES=10,CKPT_STRIDE=1 ...`.

## 2. IQM / stratified-bootstrap CI aggregation — critical path

Nothing downstream (learning curves, negative-transfer gap, the paired
E1-vs-E2 significance test promised in thesis Section 5.6) can be built
without this. No `rliable` usage or equivalent exists anywhere in the repo.

- [x] `analysis/aggregate.py` — IQM + 95% stratified-bootstrap CI (resamples
      the seed×task matrix stratified by task, not seeds alone) + PAIRED
      supervised−task_id per-(task,seed) difference test (significant iff CI
      excludes 0) + optional per-iteration curves (`--curves`). Reads wandb
      (`secondrun_*`, default) or on-cluster CSVs (`--source csv`). Stats core
      unit-tested; full wandb pull verified end-to-end.
- [ ] Read off and record the headline numbers (per-encoder IQM + the paired
      CI) once the pull finishes; drop `finger-turn-hard` retention (degenerate
      near-zero baseline) from any retention-based cut.

## 3. Context-recovery script — RQ1 evidence

Data collection is BUILT (uncommitted, see #0); analysis code still needed.

- [x] `evaluate_carl.py --probe [N]` (default 200 episodes/task): rolls out the
      **final** checkpoint only, each episode under a fresh context with every
      numeric feature perturbed **independently** within the task's feasible box
      (unlike the lockstep `--sweep`, so per-dimension R² is attributable).
      One row per episode — raw ground-truth context + ẑ snapshots at
      t=25/50/100/250 + last-100-step mean (identification-speed curve for
      free) — into `context_probe_<task>_shard<i>.csv`, with per-(ckpt, task)
      resume. `task_id` runs through the same path as the **negative control**
      (fixed per-task lookup ⇒ R² ~ 0 by construction; that contrast is itself
      a result).
- [x] `slurm/hpc/eval_carl_probe_taskid_supervised.sh`: all 6 combos
      (supervised/task_id × seeds 3,4,5), array 0-5, 8 MPS-packed episode
      shards per combo. Eval processes are wandb-free (CSV output); after all
      shards finish, each combo's CSVs are uploaded as one versioned wandb
      Artifact (`context_probe_seed<S>_<enc>`, type=dataset) in project
      `carl_context_probe`.
- [ ] Submit the probe sbatch once #0 (commit + pull on cluster) is done.

### After the probe launch — follow-up checklist

- [ ] Monitor the array: watch `eval_probe_carl_*_shard*.log` for tracebacks;
      a shard that dies mid-task leaves no partial rows (all-or-nothing per
      (ckpt, task) resume marker), so a plain resubmit of the same array
      resumes cleanly.
- [ ] Verify data landed: 6 combos × 8 tasks × ~200 episodes. Check
      `carl_context_probe` project has 6 dataset artifacts
      (`context_probe_seed<S>_<enc>`), and spot-check one CSV has the
      `z_tail_*` (64) + `z_t{25,50,100,250}_*` + `ctx_*` columns and the
      expected row count. `task_id` z-columns should be constant within a task
      (sanity check the negative control before trusting it).
- [ ] Offline analysis script (`analysis/probe_recovery.py`): **cross-
      validated** linear-probe R² per physical context dimension (per-task
      fits — pooling leaks task identity; in-sample R² with a 64-dim ẑ flatters
      the encoder, always report held-out K-fold), silhouette score on pooled
      ẑ labeled by task. Run on `z_tail` (settled embedding); optionally repeat
      per snapshot for the identification-speed curve. Emit a tidy table +
      the scatter/cluster data for #4's plot.
- [ ] Expected result to confirm: supervised R² clearly > 0 on physical dims,
      task_id R² ≈ 0 (negative control). If supervised is also ≈ 0, that is a
      real (and reportable) negative finding for RQ1 — do not paper over it.

## 4. Plotting / analysis code

Zero matplotlib/notebooks in the repo — everything below is built from
scratch once #2 and #3 exist.

- [ ] **OPEN DEPENDENCY:** single-task expert returns per mt30 task — needed
      by both curve overlays and the negative-transfer gap. Confirm whether
      per-task expert runs exist anywhere (older wandb projects?) or whether
      published TD-MPC2 single-task scores will be used; if neither, these
      runs are missing from the critical path and are expensive.
- [x] `analysis/plots.py` — IQM learning curves (E1 vs E2, 95% CI bands) and
      the per-task return bar chart (the redistribution result). dataviz-skill
      styled (blue=task_id, aqua=supervised, held stable across figures;
      validated palette; legend + direct labels). Reads aggregate.py's CSVs,
      writes PNG+PDF to `analysis/out/figs/`.
- [ ] Add the single-task-expert overlay to the learning curves. Deferred:
      `results/tdmpc2/*.csv` are the published experts but are **non-CARL**
      (unperturbed env, different y-normalization) — needs a defensible common
      normalization before overlaying, else apples-to-oranges.
- [x] Negative-transfer gap (fraction of single-task-expert ceiling retained),
      E1 vs E2 — `fig_negative_transfer`. Uses published TD-MPC2 experts
      (`results/tdmpc2/`, non-CARL: gap is indicative, not a same-harness
      ablation). Result: supervised retains a uniform ~0.43–0.53 across all 4
      walker tasks; task_id is bimodal (0.90/0.71 fwd, 0.16/0.09 back); both
      collapse to ~0 on fingers (flag: real negative transfer, or a CARL
      finger-domain harness issue? worth one check).
- [x] Zero-shot return vs. sweep magnitude, E1 vs E2 — `fig_sweep`
      (dose-response, IQM(perturbed)/IQM(baseline), fingers excluded). Both
      degrade; at max severity task_id retains 0.41 vs supervised 0.28.
      CAVEAT: mid-severity points are noisy/non-monotonic at 3 seeds (wide CIs,
      one clipped off-scale) — lean on the endpoint, not the shape.
- [x] Gradient-conflict-rate comparison, E1 vs E2 — `fig_grad_conflict`.
      CORRECTION to the earlier note above: `grad_conflict_frac` is NOT sparse
      training-time logging — `eval50k` (task_id) / `supervised_eval`
      (supervised) are dedicated re-evaluation sweeps over every saved
      checkpoint (~60 points/seed, every 50k steps), independent of the
      training run's own `eval_freq`. Restricted to seeds {3,4,5} (`eval50k`
      also has 6,7,8; `supervised_eval` has an uneven run count at seed 3/5 —
      likely resubmissions, harmless since we dedupe/aggregate per seed).
      Quantized to 1/15 steps (`grad_conflict_tasks=6` → C(6,2)=15 pairs), so
      the summary uses a last-5-checkpoint tail average, not a single point.
      Result: task_id 0.458 vs supervised 0.484, heavily overlapping CIs — **no
      significant difference**, both noisy/flat the whole run. Plotted as a
      5-point rolling mean (raw per-point markers were too dense/cluttered at
      ~60 pts/line) with the faint raw line underneath.
- [x] Held-out model-prediction error (consistency/reward), E1 vs E2 —
      `fig_model_error`, from the `secondrun_*` backfill (the one thing those
      projects are ground truth for). Result: task_id has clearly lower reward
      error (0.13 vs 0.44 at final ckpt, CIs well separated) but higher
      consistency error (0.0033 vs 0.0019, CIs barely separated) — encoders
      trade off which prediction head degrades less under context shift.
- [ ] Context-recovery scatter + clustering plot — needs #3 (probe not yet
      launched).

## 5. Lower priority

- [ ] Unit tests for the `supervised` encoder (CE and NCE paths) in
      `test_pearl_context.py` — currently only the `pearl` path is tested,
      but `supervised` is the one carrying the thesis result now.
- [ ] README.md / citation section still frame the repo as PEARL-only;
      touch up once the headline result direction is known.
- [ ] Bake `carl` into `docker/environment.yaml` instead of the
      `PYTHONPATH` injection workaround (reproducibility, not urgent).

## Parked (not on the E1-vs-E2 critical path)

- `slurm/hpc/run_ctxsweep_supervised_seed5_ampere.sh` — built and ready
  (7-task array: CE vs NCE, context_window, task_dim, context_coef). Launch
  opportunistically if spare GPU-hours show up; not blocking.
- `supervised_paramsweep` wandb project (loss-coefficient sweep, seed 5):
  as of 2026-07-07 the **baseline arm crashed at step 0 with no resumed run**
  (run `4xw5j8qv`) while conslow/horizon5/rewardup/combined are running.
  Without the baseline arm the sweep has no reference point — relaunch it if
  the sweep is to be used.
- `pearl` / `varibad` as baselines, E4 encoder-family comparison.
- `varibad` unit tests.

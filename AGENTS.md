# AGENTS.md — cTDMPC analysis & thesis handoff

Context for continuing the analysis/writeup of this project. Code structure,
git history, and CLAUDE-style build details are NOT repeated here — this file
records the non-obvious state a fresh agent needs.

## What this project is

Thesis: **mitigating negative transfer in multi-task model-based RL by replacing
explicit task IDs with an inferred context encoder**, built on TD-MPC2, evaluated
on the `mt30` DMControl suite with zero-shot generalization tested via CARL
(context-perturbed envs). The headline comparison is:

- **E1 = `task_id`** — stock TD-MPC2, learnable per-task embedding (baseline).
- **E2 = `supervised`** — a supervised context encoder inferring the task from a
  window of recent transitions (the proposed method).

`context_encoder` cfg also has `pearl` / `varibad` (unsupervised baselines, E4)
and `nce` vs `ce` context-loss variants — all parked; the critical path is
E1 vs E2 at seeds **3, 4, 5** (the protocol targets 5 seeds; only 3 exist so far).

## Analysis pipeline (all under `analysis/`, run from repo root)

Three standalone scripts, wandb-primary (usable off-cluster) with a CSV fallback
for on-cluster runs. Outputs go to `analysis/out/` (gitignored); figures to
`analysis/out/figs/` as PNG+PDF.

```bash
# 1. Aggregation -> CSVs (IQM + 95% stratified-bootstrap CIs). Each pull is
#    ~190 wandb runs and takes a few minutes; --cache reuses a parquet after.
python analysis/aggregate.py --curves --cache analysis/out/tidy_run1.parquet \
    --sweep --sweep-cache analysis/out/scen_run1.parquet \
    --grad-conflict --grad-conflict-cache analysis/out/gc_run1.parquet \
    --model-error --model-error-cache analysis/out/me_run1.parquet

# 2. Figures from those CSVs (skips any whose CSV is missing).
python analysis/plots.py --format both

# 3. RQ1 context-recovery probe analysis — NEEDS PROBE DATA (see below).
python analysis/probe_recovery.py --source wandb --cache analysis/out/probe.parquet
```

`aggregate.py` and `probe_recovery.py` have self-tested stat cores (run the
inline `python -c` tests in git history if you touch `iqm`, `stratified_bootstrap_ci`,
or `cv_r2`). `probe_recovery.py` uses **RidgeCV, not LinearRegression** — plain OLS
on the 64-dim embedding overfits a null relationship to R²=-2 (verified).

## wandb project map (entity `https-www-guc-edu-eg-`) — READ THIS

This mapping was hard to reconstruct; getting it wrong silently produces wrong
numbers. Each metric has ONE correct source project:

| metric | project(s) | why |
|---|---|---|
| **returns** (episode_reward+task[, +scn1..4]) | `taskid_generalizable`, `supervised_generalizable` | run-1: 10 episodes/ckpt, full ckpt grid. The source of truth for returns. |
| **model error** (consistency_error, reward_error) | `secondrun_taskid_generalizable`, `secondrun_supervised_generalizable` | the eval_episodes=3 / ckpt_stride=2 backfill exists ONLY for these columns; its returns are deliberately noisy — do NOT use them for returns. |
| **grad conflict** (grad_conflict_frac) | `eval50k` (task_id), `supervised_eval` (supervised) | dedicated per-checkpoint re-eval sweeps (~60 pts/seed, every 50k). NOT sparse training-time logging. `eval50k` also has seeds 6,7,8; `supervised_eval` has seed 7 and uneven counts — restrict to {3,4,5}. |
| **probe / context recovery** | `carl_context_probe` (artifacts, once probe runs) | 6 dataset artifacts `context_probe_seed<S>_<enc>`. |

Single-task **experts** for the negative-transfer gap are LOCAL CSVs at
`results/tdmpc2/<task>.csv` (published TD-MPC2 scores, `step,reward,seed`). They
are **non-CARL** (unperturbed env) — the gap is indicative, not a same-harness
ablation. Always note this caveat.

## Key findings so far (3 seeds — preliminary, CIs wide)

- **Aggregate return**: E2 IQM 17.7 [12.8,28.7] > E1 9.6 [8.0,11.4]; paired
  IQM(E2)-IQM(E1) = +8.1 [+3.4,+20.7], significant. BUT it's a **redistribution**,
  not a uniform win: E2 wins big on backward-walker tasks, E1 wins on forward
  walker, fingers ~0 for both, mean per-task delta ≈ 0.
- **Negative transfer**: E2 retains a uniform 0.43-0.53 of expert ceiling across
  walker tasks; E1 is bimodal (0.90/0.71 fwd, 0.16/0.09 back).
- **Gradient conflict**: NO significant difference (0.46 vs 0.48, overlapping) —
  does not support the RQ2 hypothesis at this seed budget. Report honestly.
- **Zero-shot dose-response**: both degrade; at max severity E1 retains slightly
  more (0.41 vs 0.28). Mid-severity noisy at 3 seeds.
- **Model error**: trade-off — E1 lower reward error (0.13 vs 0.44), E2 lower
  consistency error (0.0019 vs 0.0033).
- **Context recovery (RQ1)**: NOT YET — probe not launched.

Keep the writeup faithful: the story is nuanced (redistribution + a null on grad
conflict), not "E2 wins everywhere". Do not overclaim.

## The one open task: the RQ1 context-recovery probe

Everything is built and validated; only the data is missing. To finish RQ1:

1. **Launch the probe** (cluster; I cannot sbatch): `sbatch slurm/hpc/eval_carl_probe_taskid_supervised.sh`
   — final checkpoint only, ~200 episodes/task, every context feature perturbed
   independently, CSV-only per shard, then uploads 6 wandb dataset artifacts.
2. **Monitor / verify**: see the checklist under TODO.md §3 "After the probe launch".
3. **Analyze**: `python analysis/probe_recovery.py --source wandb --cache analysis/out/probe.parquet`
   → writes `probe_r2_median.csv`, `probe_r2_per_task.csv`, `probe_silhouette.csv`.
   Expected: E2 R² clearly > 0, task_id ≈ 0 (structural negative control). If E2
   is ALSO ≈ 0, that's a real negative RQ1 finding — report it, don't bury it.
4. **Figure**: `python analysis/plots.py` now emits `fig_context_recovery` (it
   currently prints "NO PROBE DATA yet" and skips).

## Thesis

`C:\Users\adama\Downloads\Thesis_Proposal_Adam\main.tex` (MiKTeX; `latexmk -pdf main.tex`,
biber configured). §5.5 "Preliminary Results" holds the 6 figures (in `figures/`,
copied from `analysis/out/figs/`) + a **placeholder box** `fig:res-context-recovery`
for the probe result. When the probe lands: render `fig_context_recovery.pdf`,
copy it into `figures/`, replace the `\fbox{...}` placeholder with an
`\includegraphics`, recompile.

Two things to reconcile before final submission: the results are **3 seeds** but
§5.4 protocol says **≥5**; and the redistribution / grad-conflict-null findings
should be reflected in the discussion, not just the figure captions.

## Repo notes

- Branch `unified-context`; commits from this work are pushed through `9697f13`
  plus later local commits (check `git log`). Plotting/analysis commits are not
  auto-pushed — push explicitly.
- Two stray root `*.patch` files are already-merged clutter (safe to delete;
  TODO.md §0).
- Delegation policy (antigravity plugin) is available but this analysis was all
  done inline — no subagents needed.

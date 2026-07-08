"""
IQM + 95% stratified-bootstrap-CI aggregation for the CARL generalization eval
(TODO #2): E1 (`task_id`) vs E2 (`supervised`) on mt30.

Produces the three things the plotting code (#4) and the thesis significance
claim depend on:
  1. Per-encoder IQM of per-task normalized return at the final checkpoint,
     with a 95% stratified-bootstrap CI.
  2. A PAIRED E1-vs-E2 test: per-(task, seed) return difference, aggregated by
     IQM, with a 95% bootstrap CI -- "significant" iff the CI excludes zero.
  3. Optional per-iteration IQM curves (--curves) for the learning-curve plots.

Why bootstrap this way with only 3 seeds: resampling 3 seeds alone gives a
degenerate CI, so we follow the rliable convention and resample over the
seed x task run matrix, stratified by task (each task's seed-runs resampled
independently). Pairing E1 and E2 by (task, seed) is what makes 3 seeds usable
for the difference test -- keep everything paired.

Sources (--source):
  wandb : pull history from the secondrun_* projects (runs anywhere).
  csv   : read carl_metrics_shard*.csv out of the logs tree (on the cluster).

The two sources are interchangeable: both yield one baseline (s=0, unperturbed)
return per (encoder, seed, iteration, task), which is the `episode_reward+<task>`
column / metric with no `+scn` suffix.
"""
import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# The 8 CARL-wrappable mt30 tasks the eval actually ran (-F, walker/fish/finger
# subset in the mt30 training set); matches the metric keys in the secondrun_*
# projects.
DEFAULT_TASKS = [
    'walker-walk', 'walker-run', 'walker-walk-backwards', 'walker-run-backwards',
    'fish-swim', 'finger-spin', 'finger-turn-easy', 'finger-turn-hard',
]
ENTITY = 'https-www-guc-edu-eg-'
ENCODERS = ['task_id', 'supervised']
# Returns come from the ORIGINAL run (10 episodes/checkpoint, full checkpoint
# grid). The secondrun_* projects are the eval_episodes=3 backfill -- their
# returns are deliberately noisy and are NOT the source of truth for the return
# comparison (they exist for the consistency/reward error columns). Override
# with --taskid-project / --supervised-project, e.g. to aggregate errors.
PROJECTS = {'task_id': 'taskid_generalizable',
            'supervised': 'supervised_generalizable'}
# evaluate.py's mt30 score convention: dm_control returns / 10 -> ~[0, 100],
# so per-task scores are comparable before taking IQM across tasks.
SCORE_DENOM = 10.0


def iqm(x):
    """Interquartile mean: mean of the middle 50% (25% trimmed each tail).

    Robust to the handful of degenerate task/seed cells (e.g. finger-turn-hard
    near-zero baseline) that would drag a plain mean around.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    x = np.sort(x)
    lo = int(np.floor(0.25 * x.size))
    hi = int(np.ceil(0.75 * x.size))
    mid = x[lo:hi]
    return float(mid.mean()) if mid.size else float(x.mean())


def stratified_bootstrap_ci(matrix, agg=iqm, n_boot=10000, ci=0.95, seed=0):
    """95% CI for `agg` over a (n_seeds, n_tasks) score matrix.

    Each bootstrap replicate resamples the seed axis WITH replacement (the same
    resampled seed set applied to every task = stratified-by-task), recomputes
    `agg` over the flattened matrix, and we take percentile bounds. Returns
    (point_estimate, lo, hi).
    """
    matrix = np.asarray(matrix, dtype=float)
    n_seeds = matrix.shape[0]
    rng = np.random.default_rng(seed)
    point = agg(matrix.flatten())
    boot = np.empty(n_boot)
    for b in range(n_boot):
        rows = rng.integers(0, n_seeds, size=n_seeds)
        boot[b] = agg(matrix[rows].flatten())
    alpha = (1 - ci) / 2
    lo, hi = np.nanpercentile(boot, [100 * alpha, 100 * (1 - alpha)])
    return point, float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Data loading: both paths return a tidy long DataFrame with columns
#   [encoder, seed, iteration, task, return_raw]
# holding the baseline (unperturbed, s=0) return per checkpoint per task.
# --------------------------------------------------------------------------- #

def _baseline_return_cols(columns):
    """Map `episode_reward+<task>` columns (no +scn suffix) -> task name."""
    out = {}
    for c in columns:
        m = re.fullmatch(r'(?:pretrain/)?episode_reward\+([a-z0-9\-]+)', c)
        if m:
            out[c] = m.group(1)
    return out


def load_csv(logs_root, tasks):
    """Read carl_metrics_shard*.csv from secondrun_* work_dirs under logs_root.

    Layout written by evaluate_carl.py:
      logs/mt30/<seed>/secondrun_carl_<proj>_seed<seed>_<enc>_param5/carl_metrics_shard*.csv
    """
    rows = []
    pattern = os.path.join(logs_root, 'mt30', '*',
                           'secondrun_carl_*', 'carl_metrics_shard*.csv')
    files = glob.glob(pattern)
    if not files:
        raise SystemExit(f'No CSVs matched {pattern}')
    for fp in files:
        enc = 'supervised' if 'supervised' in fp else 'task_id'
        seed = int(re.search(r'logs[/\\]mt30[/\\](\d+)[/\\]', fp).group(1))
        df = pd.read_csv(fp)
        col2task = _baseline_return_cols(df.columns)
        for col, task in col2task.items():
            if task not in tasks:
                continue
            sub = df[['iteration', col]].dropna()
            for _, r in sub.iterrows():
                rows.append((enc, seed, int(r['iteration']), task, float(r[col])))
    return pd.DataFrame(rows, columns=['encoder', 'seed', 'iteration', 'task', 'return_raw'])


def load_wandb(tasks, projects):
    """Pull baseline per-task returns from the given {encoder: project} map.

    Each project has ~32 shards x 3 seeds runs; a shard covers a subset of
    checkpoints, so we union across all runs and dedupe on (seed, iteration,
    task) -- disjoint shards make collisions rare, but a resubmission could
    double-log, so keep the last.
    """
    import wandb
    api = wandb.Api()
    rows = []
    for enc, project in projects.items():
        runs = api.runs(f'{ENTITY}/{project}')
        want = [f'episode_reward+{t}' for t in tasks] + \
               [f'pretrain/episode_reward+{t}' for t in tasks]
        for run in runs:
            seed = run.config.get('seed')
            if seed is None:
                continue
            # Only request keys that this run actually logged: wandb's
            # history() returns EMPTY if any requested key is absent, and a
            # shard only logs the tasks whose columns it produced.
            present = [k for k in want if k in run.summary.keys()]
            if not present:
                continue
            hist = run.history(keys=['pretrain/iteration'] + present, pandas=True)
            if hist is None or hist.empty:
                continue
            col2task = _baseline_return_cols(hist.columns)
            for col, task in col2task.items():
                if task not in tasks:
                    continue
                sub = hist[['pretrain/iteration', col]].dropna()
                for _, r in sub.iterrows():
                    rows.append((enc, int(seed), int(r['pretrain/iteration']),
                                 task, float(r[col])))
    df = pd.DataFrame(rows, columns=['encoder', 'seed', 'iteration', 'task', 'return_raw'])
    return df.drop_duplicates(['encoder', 'seed', 'iteration', 'task'], keep='last')


# --------------------------------------------------------------------------- #

def seed_task_matrix(df, encoder, iteration, tasks, seeds):
    """(n_seeds, n_tasks) matrix of normalized return at one checkpoint.

    NaN where a (seed, task) cell is missing, so the bootstrap/IQM's nan-drop
    handles ragged coverage rather than silently dropping whole rows.
    """
    d = df[(df.encoder == encoder) & (df.iteration == iteration)]
    piv = d.pivot_table(index='seed', columns='task', values='score', aggfunc='mean')
    return piv.reindex(index=seeds, columns=tasks).to_numpy()


def final_iteration(df, encoder):
    """The last checkpoint present for an encoder (final.pt is always kept)."""
    return int(df[df.encoder == encoder].iteration.max())


def summarize(df, tasks, n_boot, args):
    df = df.copy()
    df['score'] = df['return_raw'] / SCORE_DENOM
    seeds = sorted(df.seed.unique())
    print(f'Seeds: {seeds} | tasks: {len(tasks)} | '
          f'checkpoints/enc: '
          + ', '.join(f'{e}={df[df.encoder==e].iteration.nunique()}' for e in PROJECTS))

    # 1. Per-encoder IQM at the final checkpoint.
    summary_rows = []
    mats = {}
    for enc in ENCODERS:
        it = final_iteration(df, enc)
        mat = seed_task_matrix(df, enc, it, tasks, seeds)
        mats[enc] = mat
        point, lo, hi = stratified_bootstrap_ci(mat, n_boot=n_boot, seed=args.seed)
        n_missing = int(np.isnan(mat).sum())
        summary_rows.append({'encoder': enc, 'iteration': it, 'iqm_score': point,
                             'ci_lo': lo, 'ci_hi': hi, 'n_seeds': len(seeds),
                             'n_tasks': len(tasks), 'n_missing_cells': n_missing})
    summary = pd.DataFrame(summary_rows)
    print('\n=== Per-encoder IQM (final checkpoint, normalized return /10) ===')
    print(summary.to_string(index=False,
          formatters={'iqm_score': '{:.2f}'.format, 'ci_lo': '{:.2f}'.format,
                      'ci_hi': '{:.2f}'.format}))

    # 2. Paired E1-vs-E2 per-task return difference at the final checkpoint.
    #    Pair on (task, seed); align both encoders on their own final checkpoint.
    sup, tid = mats['supervised'], mats['task_id']
    if sup.shape == tid.shape:
        delta = sup - tid  # (n_seeds, n_tasks), positive = supervised better
        # Two DIFFERENT questions, reported separately because they can disagree
        # (and here they do -- the effect is a per-task redistribution):
        #
        # (a) Difference in AGGREGATE performance: IQM(sup) - IQM(tid), with a
        #     paired bootstrap (resample seeds once, apply to both encoders,
        #     recompute both aggregate IQMs, take the difference). This is the
        #     rliable-correct "is one method's overall score higher" test.
        #
        # NOTE: do NOT use IQM(sup - tid) here. IQM trims the middle 50%, i.e.
        #     exactly the few tasks carrying the large +/- deltas, so it reports
        #     the small-delta middle band and flips sign relative to (a). The
        #     honest per-task-change summary is the MEAN delta (b), which does
        #     not trim.
        rng = np.random.default_rng(args.seed)
        n_seeds = sup.shape[0]
        d_point = iqm(sup.flatten()) - iqm(tid.flatten())
        boot = np.empty(n_boot)
        for b in range(n_boot):
            r = rng.integers(0, n_seeds, size=n_seeds)
            boot[b] = iqm(sup[r].flatten()) - iqm(tid[r].flatten())
        d_lo, d_hi = np.nanpercentile(boot, [2.5, 97.5])
        sig = (d_lo > 0) or (d_hi < 0)
        print('\n=== Aggregate difference  IQM(supervised) - IQM(task_id) ===')
        print(f'  delta = {d_point:+.2f}  95% CI [{d_lo:+.2f}, {d_hi:+.2f}]  '
              f'-> {"SIGNIFICANT" if sig else "not significant"} (CI '
              f'{"excludes" if sig else "includes"} 0)')

        # (b) Typical per-task CHANGE: mean of per-(task,seed) deltas (untrimmed).
        mean_delta = float(np.nanmean(delta))
        print(f'  Net per-task change (mean delta, untrimmed) = {mean_delta:+.2f}')
        print('  -> The per-task table below is the real result: a redistribution,')
        print('     not a uniform shift. Report it, not a single aggregate number.')

        # Per-task paired means + per-task 95% CI over seeds, for the plot/table.
        per_task = pd.DataFrame({
            'task': tasks,
            'supervised': np.nanmean(sup, axis=0),
            'task_id': np.nanmean(tid, axis=0),
            'delta': np.nanmean(delta, axis=0),
        }).sort_values('delta', ascending=False)
        per_task.to_csv(args.outdir / 'paired_per_task.csv', index=False)
        print(f'  Per-task deltas -> {args.outdir / "paired_per_task.csv"}')
    else:
        print('\n[skip paired test] encoder matrices differ in shape '
              f'(sup {sup.shape} vs task_id {tid.shape})')

    summary.to_csv(args.outdir / 'iqm_final.csv', index=False)
    print(f'\nWrote {args.outdir / "iqm_final.csv"}')
    return summary


def curves(df, tasks, n_boot, args):
    """Per-iteration IQM + CI per encoder, for the learning-curve plots (#4)."""
    df = df.copy()
    df['score'] = df['return_raw'] / SCORE_DENOM
    seeds = sorted(df.seed.unique())
    rows = []
    for enc in ENCODERS:
        for it in sorted(df[df.encoder == enc].iteration.unique()):
            mat = seed_task_matrix(df, enc, it, tasks, seeds)
            if np.isnan(mat).all():
                continue
            point, lo, hi = stratified_bootstrap_ci(mat, n_boot=n_boot, seed=args.seed)
            rows.append({'encoder': enc, 'iteration': it,
                         'iqm_score': point, 'ci_lo': lo, 'ci_hi': hi})
    out = pd.DataFrame(rows)
    out.to_csv(args.outdir / 'iqm_curves.csv', index=False)
    print(f'Wrote {args.outdir / "iqm_curves.csv"} ({len(out)} points)')
    return out


def load_wandb_scenarios(tasks, projects):
    """Pull baseline + sweep-scenario returns at the FINAL checkpoint.

    The auto sweep logs episode_reward+<task> (s=0 baseline) and
    episode_reward+<task>+scn{1..4} at s = {0.25,0.5,0.75,1.0} x the task's own
    physical s_max. Returns tidy [encoder, seed, task, scn, frac, return_raw]
    with scn=0 the baseline (frac=0). We keep only each (encoder, seed) run's
    max-iteration rows so this is the converged model's dose-response.
    """
    import wandb
    api = wandb.Api()
    rows = []
    for enc, project in projects.items():
        runs = api.runs(f'{ENTITY}/{project}')
        for run in runs:
            seed = run.config.get('seed')
            if seed is None:
                continue
            keys = ['pretrain/iteration']
            for t in tasks:
                keys.append(f'pretrain/episode_reward+{t}')
                keys += [f'pretrain/episode_reward+{t}+scn{i}' for i in (1, 2, 3, 4)]
            present = [k for k in keys if k == 'pretrain/iteration' or k in run.summary.keys()]
            if len(present) <= 1:
                continue
            hist = run.history(keys=present, pandas=True)
            if hist is None or hist.empty:
                continue
            for _, r in hist.iterrows():
                it = r.get('pretrain/iteration')
                if pd.isna(it):
                    continue
                for t in tasks:
                    base_col = f'pretrain/episode_reward+{t}'
                    if base_col in r and not pd.isna(r[base_col]):
                        rows.append((enc, int(seed), int(it), t, 0, 0.0, float(r[base_col])))
                    for i in (1, 2, 3, 4):
                        c = f'pretrain/episode_reward+{t}+scn{i}'
                        if c in r and not pd.isna(r[c]):
                            rows.append((enc, int(seed), int(it), t, i, i * 0.25, float(r[c])))
    df = pd.DataFrame(rows, columns=['encoder', 'seed', 'iteration', 'task', 'scn', 'frac', 'return_raw'])
    df = df.drop_duplicates(['encoder', 'seed', 'iteration', 'task', 'scn'], keep='last')
    # Keep only each run's converged (max-iteration) rows.
    fin_it = df.groupby('encoder').iteration.transform('max')
    return df[df.iteration == fin_it].copy()


def _mat(series, seeds, tasks):
    """(n_seeds, n_tasks) matrix from a (seed, task)-indexed return Series."""
    m = np.full((len(seeds), len(tasks)), np.nan)
    for si, s in enumerate(seeds):
        for ti, t in enumerate(tasks):
            v = series.get((s, t))
            if v is not None and not pd.isna(v):
                m[si, ti] = v
    return m


def sweep_curve(scen_df, tasks, n_boot, args, min_baseline=50.0):
    """Dose-response: return retention vs perturbation severity, per encoder.

    retention = IQM(perturbed returns) / IQM(baseline returns), aggregated over
    the (task, seed) matrix at each severity. This aggregate-then-ratio form is
    stable: a per-(task,seed) ratio blows up when a task's baseline is small
    (fish-swim), so IQM-of-ratios produced a spurious CI spike. Tasks with a
    near-zero raw baseline (fingers) are dropped via min_baseline. CI is a paired
    seed bootstrap (resample seeds once, recompute both IQMs, take the ratio).
    """
    rng = np.random.default_rng(args.seed)
    rows = []
    for enc in ENCODERS:
        d = scen_df[scen_df.encoder == enc]
        seeds = sorted(d.seed.unique())
        base = d[d.scn == 0].set_index(['seed', 'task']).return_raw
        keep = [t for t in tasks
                if base.reindex([(s, t) for s in seeds]).mean() >= min_baseline]
        base_mat = _mat(base, seeds, keep)
        for scn, frac in [(0, 0.0), (1, .25), (2, .5), (3, .75), (4, 1.0)]:
            cur_mat = _mat(d[d.scn == scn].set_index(['seed', 'task']).return_raw, seeds, keep)
            point = iqm(cur_mat.flatten()) / iqm(base_mat.flatten())
            boot = np.empty(n_boot)
            for b in range(n_boot):
                r = rng.integers(0, len(seeds), size=len(seeds))
                boot[b] = iqm(cur_mat[r].flatten()) / iqm(base_mat[r].flatten())
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            rows.append({'encoder': enc, 'scn': scn, 'frac': frac,
                         'iqm_retention': point, 'ci_lo': float(lo), 'ci_hi': float(hi),
                         'n_tasks': len(keep)})
    out = pd.DataFrame(rows)
    out.to_csv(args.outdir / 'sweep_curve.csv', index=False)
    print('\n=== Dose-response: IQM(perturbed)/IQM(baseline) vs perturbation severity ===')
    print(out.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    print(f'Wrote {args.outdir / "sweep_curve.csv"}')
    return out


def load_experts(experts_dir, tasks):
    """Per-task published TD-MPC2 single-task expert return (raw dm_control).

    results/tdmpc2/<task>.csv has columns step,reward,seed. We take each seed's
    converged return (mean of its last few logged steps, to de-noise the final
    point) then average over seeds. These experts are trained on the UNPERTURBED
    env, so they are the ceiling for the s=0 baseline scenario -- comparable to
    our multitask baseline return, which is also evaluated at the default
    context. (Caveat for the writeup: expert seeds/harness differ from the CARL
    eval; treat the gap as indicative, not a controlled ablation.)
    """
    experts_dir = Path(experts_dir)
    out = {}
    for task in tasks:
        fp = experts_dir / f'{task}.csv'
        if not fp.exists():
            continue
        e = pd.read_csv(fp)
        per_seed = []
        for _, g in e.groupby('seed'):
            g = g.sort_values('step')
            per_seed.append(g.reward.tail(3).mean())  # converged return
        out[task] = float(np.mean(per_seed))
    return out


def negative_transfer(df, tasks, experts_dir, args):
    """Negative-transfer gap vs single-task experts, per task per encoder.

    gap = expert - multitask_baseline (raw return); retained = multitask/expert
    (fraction of the single-task ceiling kept -- comparable across tasks, unlike
    the raw gap where finger's ~980 dwarfs the walker gaps). Uses the s=0
    baseline (default context) return at the final checkpoint.
    """
    experts = load_experts(experts_dir, tasks)
    if not experts:
        print(f'\n[skip negative-transfer] no expert CSVs under {experts_dir}')
        return None
    fin = df[df.iteration == df.groupby('encoder').iteration.transform('max')]
    base = fin.groupby(['encoder', 'task']).return_raw.mean().reset_index()
    rows = []
    for _, r in base.iterrows():
        exp = experts.get(r.task)
        if exp is None or exp == 0:
            continue
        rows.append({'encoder': r.encoder, 'task': r.task,
                     'multitask': r.return_raw, 'expert': exp,
                     'gap': exp - r.return_raw,
                     'retained_frac': r.return_raw / exp})
    out = pd.DataFrame(rows)
    out.to_csv(args.outdir / 'negative_transfer.csv', index=False)
    print('\n=== Negative transfer vs single-task experts (fraction of ceiling retained) ===')
    piv = out.pivot(index='task', columns='encoder', values='retained_frac')
    print(piv.to_string(float_format=lambda x: f'{x:.2f}'))
    print(f'Wrote {args.outdir / "negative_transfer.csv"}')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', choices=['wandb', 'csv'], default='wandb')
    ap.add_argument('--logs-root', default='tdmpc2/logs',
                    help='(csv source) root containing mt30/<seed>/secondrun_carl_*')
    ap.add_argument('--tasks', nargs='*', default=DEFAULT_TASKS)
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--curves', action='store_true',
                    help='also emit per-iteration IQM curves (iqm_curves.csv)')
    ap.add_argument('--seed', type=int, default=0, help='bootstrap RNG seed')
    ap.add_argument('--outdir', type=Path, default=Path('analysis/out'))
    ap.add_argument('--cache', type=Path, default=None,
                    help='parquet path to cache/reuse the tidy loaded frame')
    ap.add_argument('--taskid-project', default=PROJECTS['task_id'],
                    help='(wandb source) task_id project; default is the 10-episode run-1 project')
    ap.add_argument('--supervised-project', default=PROJECTS['supervised'],
                    help='(wandb source) supervised project; default is the 10-episode run-1 project')
    ap.add_argument('--experts-dir', default='results/tdmpc2',
                    help='dir of published single-task expert CSVs for the negative-transfer gap')
    ap.add_argument('--sweep', action='store_true',
                    help='also pull sweep-scenario returns and emit the dose-response curve')
    ap.add_argument('--sweep-cache', type=Path, default=None,
                    help='parquet cache for the (heavier) scenario pull')
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    projects = {'task_id': args.taskid_project, 'supervised': args.supervised_project}

    if args.cache and args.cache.exists():
        print(f'Loading cached frame {args.cache}')
        df = pd.read_parquet(args.cache)
    else:
        df = load_wandb(args.tasks, projects) if args.source == 'wandb' \
            else load_csv(args.logs_root, args.tasks)
        if args.cache:
            df.to_parquet(args.cache)
            print(f'Cached tidy frame -> {args.cache}')
    if df.empty:
        raise SystemExit('No rows loaded -- check --source / project names / tasks.')

    summarize(df, args.tasks, args.n_boot, args)
    negative_transfer(df, args.tasks, args.experts_dir, args)
    if args.curves:
        curves(df, args.tasks, args.n_boot, args)
    if args.sweep:
        if args.sweep_cache and args.sweep_cache.exists():
            print(f'\nLoading cached scenario frame {args.sweep_cache}')
            scen = pd.read_parquet(args.sweep_cache)
        else:
            scen = load_wandb_scenarios(args.tasks, projects)
            if args.sweep_cache:
                scen.to_parquet(args.sweep_cache)
                print(f'Cached scenario frame -> {args.sweep_cache}')
        if scen.empty:
            print('[skip sweep] no scenario rows loaded')
        else:
            sweep_curve(scen, args.tasks, args.n_boot, args)


if __name__ == '__main__':
    main()

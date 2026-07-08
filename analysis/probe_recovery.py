"""
RQ1 context-recovery analysis (TODO #3): does the inferred context embedding
ẑ track the real physical parameters CARL used to generate the episode, or
does it only memorize which of the 30 discrete tasks it's in?

Reads the CSVs written by `evaluate_carl.py --probe` (one row per episode:
z_tail_<0..63> + z_t{25,50,100,250}_<i> + ctx_<feature> + task/episode/reward),
either from local files or from the `carl_context_probe` wandb artifacts.

Two analyses, both PER TASK (pooling would leak task identity into the probe --
see TODO note) and both CROSS-VALIDATED (in-sample R^2 with a 64-dim ẑ would
flatter the encoder regardless of whether it generalizes):

  1. Linear-probe R^2 per physical context dimension: K-fold CV, ẑ_tail -> each
     ctx_<feature> independently. High R^2 = the embedding linearly encodes
     that physical quantity despite never being supervised on it.
  2. Silhouette score on pooled ẑ_tail (across tasks, within an encoder+seed),
     labeled by task -- unsupervised clustering-quality check, complementary
     to the (supervised) probe accuracy already seen during training.

task_id is the NEGATIVE CONTROL: its embedding is a fixed per-task lookup, so
every episode of a task gets an IDENTICAL ẑ regardless of the sampled physical
context -- R^2 must be ~0 (or an sklearn warning about constant predictions/
undefined R^2) by construction. If task_id's probe methodology can't produce
that null result, don't trust the numbers for supervised either.
"""
import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold

# Ridge, not plain LinearRegression: ẑ is 64-dim and a task's episode budget
# (default 200) leaves only ~160 rows per CV training fold -- OLS on p=64,
# n~160 overfits badly on a NULL relationship (verified: an unrelated feature
# gave R^2 = -2.1 with plain LinearRegression in this file's self-test).
# RidgeCV picks its regularization strength per fold via internal CV, which
# keeps a genuine signal's R^2 high while pulling a null relationship back
# toward ~0 instead of deeply negative.
RIDGE_ALPHAS = np.logspace(-2, 4, 13)

ENTITY = 'https-www-guc-edu-eg-'
SEEDS = (3, 4, 5)
ENCODERS = ('task_id', 'supervised')


def load_local(logs_root, embed_col_prefix='z_tail_'):
    """Read context_probe_<task>_shard*.csv from probe_carl_* work_dirs.

    Layout written by evaluate_carl.py --probe:
      logs/mt30/<seed>/probe_carl_seed<seed>_<enc>_param5/context_probe_<task>_shard*.csv
    """
    pattern = str(Path(logs_root) / 'mt30' / '*' / 'probe_carl_*' / 'context_probe_*_shard*.csv')
    files = glob.glob(pattern)
    if not files:
        raise SystemExit(f'No CSVs matched {pattern}')
    frames = []
    for fp in files:
        m = re.search(r'[/\\](\d+)[/\\]probe_carl_seed\d+_([a-z_]+)_param5[/\\]', fp)
        if not m:
            continue
        seed, enc = int(m.group(1)), m.group(2)
        df = pd.read_csv(fp)
        df['seed'] = seed
        df['encoder'] = enc
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_wandb_artifacts(download_dir):
    """Download the 6 context_probe_seed<S>_<enc> dataset artifacts and
    concatenate their CSVs, tagging each row with (seed, encoder).
    """
    import wandb
    api = wandb.Api()
    frames = []
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    for enc in ENCODERS:
        for seed in SEEDS:
            name = f'{ENTITY}/carl_context_probe/context_probe_seed{seed}_{enc}:latest'
            try:
                art = api.artifact(name, type='dataset')
            except Exception as e:
                print(f'[skip] {name}: {e}')
                continue
            local = art.download(root=str(download_dir / f'{enc}_{seed}'))
            for fp in glob.glob(str(Path(local) / 'context_probe_*.csv')):
                df = pd.read_csv(fp)
                df['seed'] = seed
                df['encoder'] = enc
                frames.append(df)
    if not frames:
        raise SystemExit('No probe artifacts found -- has the probe sbatch finished/uploaded?')
    return pd.concat(frames, ignore_index=True)


def embedding_matrix(df, prefix):
    cols = sorted([c for c in df.columns if c.startswith(prefix)],
                 key=lambda c: int(c[len(prefix):]))
    if not cols:
        raise SystemExit(f'No embedding columns with prefix {prefix!r} found -- '
                         f'check the probe CSV schema.')
    return df[cols].to_numpy(), cols


def context_feature_cols(df):
    return [c for c in df.columns if c.startswith('ctx_')]


def cv_r2(X, y, n_splits=5, seed=0):
    """K-fold cross-validated R^2 of a linear probe X -> y.

    Returns nan if y is (near-)constant in a fold (undefined R^2 -- expected
    for task_id, where every row in a task has an identical y... no, y is the
    physical context, which DOES vary for task_id too; it's X, ẑ, that's
    constant for task_id. A constant X makes the fit degenerate -> R^2 ~ 0 or
    undefined, which IS the expected negative-control result, not an error).
    """
    n = len(y)
    if n < n_splits or np.nanstd(y) == 0:
        return np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.full(n, np.nan)
    for train_idx, test_idx in kf.split(X):
        if np.nanstd(X[train_idx]) == 0:
            preds[test_idx] = y[train_idx].mean()  # degenerate fit -> predict the mean
            continue
        model = RidgeCV(alphas=RIDGE_ALPHAS).fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])
    ss_res = np.nansum((y - preds) ** 2)
    ss_tot = np.nansum((y - np.nanmean(y)) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1 - ss_res / ss_tot)


def probe_r2(df, embed_prefix, n_splits, seed, min_rows=20):
    """Per-task, per-encoder, per-context-dimension cross-validated R^2."""
    ctx_cols = context_feature_cols(df)
    rows = []
    for enc in ENCODERS:
        for task, td in df[df.encoder == enc].groupby('task'):
            if len(td) < min_rows:
                print(f'  [skip] {enc}/{task}: only {len(td)} rows (< {min_rows})')
                continue
            X, _ = embedding_matrix(td, embed_prefix)
            for ctx_col in ctx_cols:
                y = td[ctx_col].to_numpy(dtype=float)
                if np.all(np.isnan(y)):
                    continue
                mask = ~np.isnan(y)
                if mask.sum() < min_rows:
                    continue
                r2 = cv_r2(X[mask], y[mask], n_splits=n_splits, seed=seed)
                rows.append({'encoder': enc, 'task': task,
                            'context_feature': ctx_col.removeprefix('ctx_'),
                            'r2_cv': r2, 'n': int(mask.sum())})
    return pd.DataFrame(rows)


def silhouette_by_task(df, embed_prefix, seed, min_rows=20):
    """Silhouette score on pooled ẑ (across tasks) labeled by task, per
    encoder+seed. For task_id this should be ~1 for a trivial reason (each
    task's ẑ is a single point, so clusters are maximally separated) -- the
    interesting comparison is whether supervised achieves comparable
    separation despite ẑ actually varying within a task.
    """
    rows = []
    for enc in ENCODERS:
        for s, sd in df[df.encoder == enc].groupby('seed'):
            counts = sd.task.value_counts()
            keep_tasks = counts[counts >= min_rows].index
            sd = sd[sd.task.isin(keep_tasks)]
            if sd.task.nunique() < 2:
                continue
            X, _ = embedding_matrix(sd, embed_prefix)
            labels = sd.task.to_numpy()
            score = silhouette_score(X, labels, random_state=seed)
            rows.append({'encoder': enc, 'seed': int(s), 'silhouette': float(score),
                        'n_tasks': sd.task.nunique(), 'n_rows': len(sd)})
    return pd.DataFrame(rows)


def summarize_r2(r2_df, outdir):
    if r2_df.empty:
        print('[skip] no R^2 rows computed')
        return
    r2_df.to_csv(outdir / 'probe_r2_per_task.csv', index=False)
    agg = r2_df.groupby(['encoder', 'context_feature']).r2_cv.median().reset_index()
    agg = agg.pivot(index='context_feature', columns='encoder', values='r2_cv')
    print('\n=== Median cross-validated R^2 per physical dimension (across tasks) ===')
    print(agg.to_string(float_format=lambda x: f'{x:.3f}'))
    agg.to_csv(outdir / 'probe_r2_median.csv')
    print(f'\nWrote {outdir / "probe_r2_per_task.csv"} and {outdir / "probe_r2_median.csv"}')
    print('\nExpected: supervised R^2 clearly > 0, task_id R^2 ~= 0 (negative control).')
    print('If supervised is ALSO ~0, that is a real negative RQ1 finding -- report it, do not bury it.')


def summarize_silhouette(sil_df, outdir):
    if sil_df.empty:
        print('[skip] no silhouette rows computed')
        return
    sil_df.to_csv(outdir / 'probe_silhouette.csv', index=False)
    print('\n=== Silhouette score (ẑ pooled across tasks, labeled by task) ===')
    print(sil_df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print(f'Wrote {outdir / "probe_silhouette.csv"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', choices=['wandb', 'csv'], default='wandb')
    ap.add_argument('--logs-root', default='tdmpc2/logs',
                    help='(csv source) root containing mt30/<seed>/probe_carl_*')
    ap.add_argument('--download-dir', default='analysis/out/probe_artifacts',
                    help='(wandb source) where to download artifact CSVs')
    ap.add_argument('--embed', choices=['tail', 't25', 't50', 't100', 't250'], default='tail',
                    help='which ẑ snapshot to probe (default: settled last-100-step mean)')
    ap.add_argument('--n-splits', type=int, default=5)
    ap.add_argument('--min-rows', type=int, default=20,
                    help='minimum episodes required for a (task, encoder) probe fit')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--outdir', type=Path, default=Path('analysis/out'))
    ap.add_argument('--cache', type=Path, default=None)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    embed_prefix = 'z_tail_' if args.embed == 'tail' else f'z_{args.embed}_'

    if args.cache and args.cache.exists():
        print(f'Loading cached probe frame {args.cache}')
        df = pd.read_parquet(args.cache)
    else:
        df = load_wandb_artifacts(args.download_dir) if args.source == 'wandb' \
            else load_local(args.logs_root)
        if args.cache:
            df.to_parquet(args.cache)
            print(f'Cached probe frame -> {args.cache}')
    if df.empty:
        raise SystemExit('No probe rows loaded.')
    print(f'Loaded {len(df)} episode rows: encoders={sorted(df.encoder.unique())}, '
         f'seeds={sorted(df.seed.unique())}, tasks={df.task.nunique()}')

    r2_df = probe_r2(df, embed_prefix, args.n_splits, args.seed, args.min_rows)
    summarize_r2(r2_df, args.outdir)

    sil_df = silhouette_by_task(df, embed_prefix, args.seed, args.min_rows)
    summarize_silhouette(sil_df, args.outdir)


if __name__ == '__main__':
    main()

"""
Publication figures for the CARL generalization comparison (TODO #4), built from
the aggregate.py outputs (analysis/out/*.csv).

Six figures, each rendered only if its input CSV exists (run the matching
aggregate.py flag first -- see each fig_* function's docstring for which):
  A. fig_learning_curves    IQM return over training, 95% CI bands.
  B. fig_per_task           per-task return at the final checkpoint -- the
                            redistribution result (supervised lifts the
                            backward-walker tasks, cedes the forward ones).
  C. fig_negative_transfer  fraction of the published single-task-expert
                            ceiling retained, per task. Caveat baked into
                            aggregate.py: experts are non-CARL runs (different
                            seeds/harness), so this gap is indicative, not a
                            controlled same-harness ablation.
  D. fig_sweep              dose-response (return retention) vs perturbation
                            severity; y-axis capped with off-scale CI
                            annotated (3-seed mid-severity noise).
  E. fig_grad_conflict      cross-task gradient-conflict fraction over
                            training, from the eval50k/supervised_eval
                            projects (dense, every-50k-checkpoint re-eval, NOT
                            the sparse training-time log). Quantized to 1/15
                            steps (grad_conflict_tasks=6 -> C(6,2)=15 pairs).
  F. fig_model_error        held-out consistency_error / reward_error under
                            CARL context shift, from the secondrun_* backfill
                            (the one thing those projects are ground truth for).
  G. fig_context_recovery   RQ1: median cross-validated linear-probe R^2 of ẑ
                            -> each physical context dim, supervised vs the
                            task_id negative control (from probe_recovery.py).
                            SKIPPED until the probe (TODO #3) has been run --
                            no data exists yet.

Design follows the dataviz skill: categorical hues assigned by ENTITY and held
stable across all figures (blue=task_id, aqua=supervised), thin marks, recessive
axes/grid, text in ink tokens (never the series color), a legend plus direct
labels (aqua is sub-3:1 on the light surface, so the relief rule requires
visible labels -- and the CSVs are the table view).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- palette (dataviz reference instance, light mode) -----------------------
# Categorical slots assigned by entity and never cycled/repainted.
C = {
    'task_id':    '#2a78d6',   # slot 1 blue
    'supervised': '#1baf7a',   # slot 2 aqua
    'surface':    '#fcfcfb',
    'ink':        '#0b0b0b',   # primary text
    'ink2':       '#52514e',   # secondary text
    'muted':      '#898781',   # axis / tick labels
    'grid':       '#e1e0d9',   # hairline gridline
    'axis':       '#c3c2b7',   # baseline / axis
}
LABEL = {'task_id': 'task_id (E1)', 'supervised': 'supervised (E2)'}


def setup_style():
    plt.rcParams.update({
        'figure.facecolor': C['surface'],
        'axes.facecolor': C['surface'],
        'savefig.facecolor': C['surface'],
        'font.family': ['DejaVu Sans'],
        'font.size': 10,
        'axes.edgecolor': C['axis'],
        'axes.linewidth': 0.8,
        'axes.labelcolor': C['ink2'],
        'axes.titlecolor': C['ink'],
        'xtick.color': C['muted'],
        'ytick.color': C['muted'],
        'xtick.labelcolor': C['ink2'],
        'ytick.labelcolor': C['ink2'],
        'axes.grid': True,
        'grid.color': C['grid'],
        'grid.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def _millions(x, _pos):
    if x == 0:
        return '0'
    return f'{x/1e6:.1f}M'.replace('.0M', 'M')


def _save(fig, out_stem, fmt):
    outs = []
    for ext in (['png', 'pdf'] if fmt == 'both' else [fmt]):
        fp = f'{out_stem}.{ext}'
        fig.savefig(fp, dpi=200, bbox_inches='tight')
        outs.append(fp)
    plt.close(fig)
    return outs


def fig_learning_curves(curves_csv, out_stem, fmt):
    df = pd.read_csv(curves_csv)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for enc in ('task_id', 'supervised'):
        d = df[df.encoder == enc].sort_values('iteration')
        if d.empty:
            continue
        ax.fill_between(d.iteration, d.ci_lo, d.ci_hi, color=C[enc], alpha=0.15,
                        linewidth=0)
        ax.plot(d.iteration, d.iqm_score, color=C[enc], linewidth=2, label=LABEL[enc])
        # Direct label at the last point (relief for the sub-3:1 aqua series).
        last = d.iloc[-1]
        ax.annotate(LABEL[enc], (last.iteration, last.iqm_score),
                    xytext=(6, 0), textcoords='offset points', va='center',
                    color=C[enc], fontsize=9, fontweight='bold')
    ax.xaxis.set_major_formatter(FuncFormatter(_millions))
    ax.set_xlabel('training iteration')
    ax.set_ylabel('IQM normalized return (÷10)')
    ax.set_title('CARL generalization: IQM return over training  (95% bootstrap CI)',
                 fontsize=11, loc='left', pad=10)
    ax.margins(x=0.02)
    ax.set_xlim(left=0)
    # Legend present for >=2 series (identity never color-alone), kept recessive.
    leg = ax.legend(loc='upper left', frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(C['ink2'])
    fig.tight_layout()
    return _save(fig, out_stem, fmt)


def fig_per_task(pertask_csv, out_stem, fmt):
    df = pd.read_csv(pertask_csv).sort_values('delta')  # ascending -> biggest
    tasks = df.task.tolist()                             # supervised win at top
    y = np.arange(len(tasks))
    h = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.barh(y + h/2 + 0.02, df.supervised, height=h, color=C['supervised'],
            label=LABEL['supervised'])
    ax.barh(y - h/2 - 0.02, df.task_id, height=h, color=C['task_id'],
            label=LABEL['task_id'])
    # Direct value labels (relief + informative).
    xmax = max(df.supervised.max(), df.task_id.max())
    for yi, v in zip(y + h/2 + 0.02, df.supervised):
        ax.text(v + xmax*0.01, yi, f'{v:.0f}', va='center', ha='left',
                color=C['ink2'], fontsize=8)
    for yi, v in zip(y - h/2 - 0.02, df.task_id):
        ax.text(v + xmax*0.01, yi, f'{v:.0f}', va='center', ha='left',
                color=C['ink2'], fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(tasks, color=C['ink2'], fontsize=9)
    ax.set_xlabel('normalized return (÷10) at final checkpoint')
    ax.set_title('Per-task return, supervised vs task_id  (redistribution, not a uniform shift)',
                 fontsize=11, loc='left', pad=10)
    ax.grid(axis='y', visible=False)
    ax.margins(x=0.08)
    ax.set_xlim(left=0)
    # Park the legend in the near-empty middle-right band (the finger tasks are
    # ~0, so that column is whitespace) to avoid colliding with the long bars.
    leg = ax.legend(loc='center right', frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(C['ink2'])
    fig.tight_layout()
    return _save(fig, out_stem, fmt)


def fig_nt_gap(nt_csv, out_stem, fmt):
    df = pd.read_csv(nt_csv)
    piv = df.pivot(index='task', columns='encoder', values='retained_frac')
    piv['delta'] = piv['supervised'] - piv['task_id']
    piv = piv.sort_values('delta')  # supervised-favored at top
    tasks = piv.index.tolist()
    y = np.arange(len(tasks))
    h = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.barh(y + h/2 + 0.02, piv['supervised'], height=h, color=C['supervised'],
            label=LABEL['supervised'])
    ax.barh(y - h/2 - 0.02, piv['task_id'], height=h, color=C['task_id'],
            label=LABEL['task_id'])
    for yi, v in zip(y + h/2 + 0.02, piv['supervised']):
        ax.text(v + 0.012, yi, f'{v:.2f}', va='center', ha='left', color=C['ink2'], fontsize=8)
    for yi, v in zip(y - h/2 - 0.02, piv['task_id']):
        ax.text(v + 0.012, yi, f'{v:.2f}', va='center', ha='left', color=C['ink2'], fontsize=8)
    # Expert ceiling at 1.0 -- the gap to this line IS the negative transfer.
    ax.axvline(1.0, color=C['muted'], linewidth=1, linestyle='--')
    ax.text(1.0, len(tasks) - 0.3, ' single-task expert', color=C['muted'],
            fontsize=8, va='top', ha='left')
    ax.set_yticks(y)
    ax.set_yticklabels(tasks, color=C['ink2'], fontsize=9)
    ax.set_xlabel('fraction of single-task-expert return retained (1.0 = no negative transfer)')
    ax.set_title('Negative transfer: multitask return vs single-task ceiling',
                 fontsize=11, loc='left', pad=10)
    ax.grid(axis='y', visible=False)
    ax.set_xlim(0, 1.12)
    leg = ax.legend(loc='center right', frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(C['ink2'])
    fig.tight_layout()
    return _save(fig, out_stem, fmt)


def fig_sweep(sweep_csv, out_stem, fmt):
    df = pd.read_csv(sweep_csv)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for enc in ('task_id', 'supervised'):
        d = df[df.encoder == enc].sort_values('frac')
        if d.empty:
            continue
        ax.fill_between(d.frac, d.ci_lo, d.ci_hi, color=C[enc], alpha=0.15, linewidth=0)
        ax.plot(d.frac, d.iqm_retention, color=C[enc], linewidth=2,
                marker='o', markersize=5, label=LABEL[enc])
        last = d.iloc[-1]
        ax.annotate(LABEL[enc], (last.frac, last.iqm_retention),
                    xytext=(6, 0), textcoords='offset points', va='center',
                    color=C[enc], fontsize=9, fontweight='bold')
    ax.axhline(1.0, color=C['muted'], linewidth=1, linestyle='--')
    ax.set_xlabel("perturbation severity  (fraction of each task's feasible context range)")
    ax.set_ylabel('IQM return retention (perturbed ÷ baseline)')
    ax.set_title('Dose-response under context shift  (95% bootstrap CI; fingers excluded)',
                 fontsize=11, loc='left', pad=10)
    ax.set_xlim(0, 1.15)
    # Cap y: with 3 seeds the mid-severity CI is very wide (a resampled baseline
    # IQM can be small, inflating the ratio). Annotate any band clipped off-scale
    # rather than let it dominate; the robust signal is the max-severity endpoint.
    ytop = 1.6
    ax.set_ylim(0.15, ytop)
    for _, r in df.iterrows():
        if r.ci_hi > ytop:
            ax.annotate(f'CI→{r.ci_hi:.1f}', (r.frac, ytop), xytext=(0, -2),
                        textcoords='offset points', ha='center', va='top',
                        color=C['muted'], fontsize=7)
    leg = ax.legend(loc='lower left', frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(C['ink2'])
    fig.tight_layout()
    return _save(fig, out_stem, fmt)


def fig_grad_conflict(curve_csv, out_stem, fmt, smooth_window=5):
    """Line only, no per-point markers: at ~60 points/line (every 50k steps)
    markers add clutter, not information -- matches fig_learning_curves. The
    raw signal is quantized (1/15 steps) and genuinely noisy (no trend, no
    encoder gap -- see grad_conflict_summary's tail-average test), so a light
    rolling mean is drawn on top of the raw (faint) line to make the "flat and
    overlapping" read honest rather than looking like noise-hunting.
    """
    df = pd.read_csv(curve_csv)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for enc in ('task_id', 'supervised'):
        d = df[df.encoder == enc].sort_values('iteration')
        if d.empty:
            continue
        ax.plot(d.iteration, d.iqm_grad_conflict_frac, color=C[enc], linewidth=1,
                alpha=0.35)
        smoothed = d.iqm_grad_conflict_frac.rolling(smooth_window, center=True,
                                                     min_periods=1).mean()
        ax.fill_between(d.iteration, d.ci_lo, d.ci_hi, color=C[enc], alpha=0.10, linewidth=0)
        ax.plot(d.iteration, smoothed, color=C[enc], linewidth=2.2, label=LABEL[enc])
        last = d.iloc[-1]
        ax.annotate(LABEL[enc], (last.iteration, smoothed.iloc[-1]),
                    xytext=(6, 0), textcoords='offset points', va='center',
                    color=C[enc], fontsize=9, fontweight='bold')
    ax.xaxis.set_major_formatter(FuncFormatter(_millions))
    ax.set_xlabel('training iteration')
    ax.set_ylabel('gradient-conflict fraction (IQM over seeds)')
    ax.set_title(f'Cross-task gradient conflict over training  ({smooth_window}-pt rolling mean; '
                 'quantized to 1/15 steps)', fontsize=11, loc='left', pad=10)
    ax.set_xlim(left=0)
    ax.margins(y=0.15)
    leg = ax.legend(loc='upper left', frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(C['ink2'])
    fig.tight_layout()
    return _save(fig, out_stem, fmt)


def fig_model_error(curve_csv, out_stem, fmt):
    df = pd.read_csv(curve_csv)
    metrics = [m for m in ('consistency_error', 'reward_error') if (df.metric == m).any()]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.5 * len(metrics), 4.2))
    axes = np.atleast_1d(axes)
    titles = {'consistency_error': 'Consistency error (‖ẑ_pred − ẑ_true‖²)',
             'reward_error': 'Reward error ((r̂ − r)²)'}
    for ax, metric in zip(axes, metrics):
        md = df[df.metric == metric]
        for enc in ('task_id', 'supervised'):
            d = md[md.encoder == enc].sort_values('iteration')
            if d.empty:
                continue
            ax.fill_between(d.iteration, d.ci_lo, d.ci_hi, color=C[enc], alpha=0.15, linewidth=0)
            ax.plot(d.iteration, d.iqm_error, color=C[enc], linewidth=2, label=LABEL[enc])
            last = d.iloc[-1]
            ax.annotate(LABEL[enc], (last.iteration, last.iqm_error),
                        xytext=(6, 0), textcoords='offset points', va='center',
                        color=C[enc], fontsize=9, fontweight='bold')
        ax.xaxis.set_major_formatter(FuncFormatter(_millions))
        ax.set_xlabel('training iteration')
        ax.set_ylabel('IQM error (lower = better)')
        ax.set_title(titles.get(metric, metric), fontsize=10, loc='left')
        ax.set_xlim(left=0)
        leg = ax.legend(loc='upper right', frameon=False, fontsize=8)
        for t in leg.get_texts():
            t.set_color(C['ink2'])
    fig.suptitle('Held-out model-prediction error under CARL context shift',
                 fontsize=11, x=0.02, ha='left', y=1.02)
    fig.tight_layout()
    return _save(fig, out_stem, fmt)


def fig_context_recovery(r2_csv, out_stem, fmt):
    """RQ1 headline (from probe_recovery.py): median cross-validated linear-probe
    R^2 of ẑ -> each physical context dimension, supervised vs task_id. task_id
    is the negative control (fixed per-task embedding -> R^2 ~ 0 by construction);
    a supervised bar clearly above 0 is the evidence that the context encoder
    recovers physical structure it was never supervised on.

    Input probe_r2_median.csv: index=context_feature, columns per encoder.
    """
    df = pd.read_csv(r2_csv, index_col=0)
    have = [e for e in ('task_id', 'supervised') if e in df.columns]
    df = df.sort_values(have[-1] if have else df.columns[-1], ascending=True)
    feats = df.index.tolist()
    y = np.arange(len(feats))
    h = 0.38
    fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.5 * len(feats) + 1.5)))
    if 'supervised' in df.columns:
        ax.barh(y + h/2 + 0.02, df['supervised'].clip(lower=0), height=h,
                color=C['supervised'], label=LABEL['supervised'])
    if 'task_id' in df.columns:
        ax.barh(y - h/2 - 0.02, df['task_id'].clip(lower=0), height=h,
                color=C['task_id'], label=LABEL['task_id'])
    for enc, off in (('supervised', h/2 + 0.02), ('task_id', -h/2 - 0.02)):
        if enc in df.columns:
            for yi, v in zip(y + off, df[enc]):
                ax.text(max(v, 0) + 0.01, yi, f'{v:.2f}', va='center', ha='left',
                        color=C['ink2'], fontsize=8)
    ax.axvline(0.0, color=C['axis'], linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(feats, color=C['ink2'], fontsize=9)
    ax.set_xlabel('median cross-validated linear-probe R² (ẑ → physical context dim)')
    ax.set_title('Context recovery: does ẑ encode the physical context it was never trained on?',
                 fontsize=11, loc='left', pad=10)
    ax.grid(axis='y', visible=False)
    ax.set_xlim(left=min(0, float(df.min().min()) - 0.02))
    leg = ax.legend(loc='lower right', frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(C['ink2'])
    fig.tight_layout()
    return _save(fig, out_stem, fmt)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--indir', type=Path, default=Path('analysis/out'))
    ap.add_argument('--outdir', type=Path, default=Path('analysis/out/figs'))
    ap.add_argument('--format', choices=['png', 'pdf', 'both'], default='both')
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    setup_style()

    made = []
    curves = args.indir / 'iqm_curves.csv'
    if curves.exists():
        made += fig_learning_curves(curves, str(args.outdir / 'fig_learning_curves'), args.format)
    else:
        print(f'[skip] {curves} not found -- run aggregate.py --curves first')
    pertask = args.indir / 'paired_per_task.csv'
    if pertask.exists():
        made += fig_per_task(pertask, str(args.outdir / 'fig_per_task'), args.format)
    else:
        print(f'[skip] {pertask} not found -- run aggregate.py first')
    nt = args.indir / 'negative_transfer.csv'
    if nt.exists():
        made += fig_nt_gap(nt, str(args.outdir / 'fig_negative_transfer'), args.format)
    else:
        print(f'[skip] {nt} not found -- run aggregate.py first')
    sweep = args.indir / 'sweep_curve.csv'
    if sweep.exists():
        made += fig_sweep(sweep, str(args.outdir / 'fig_sweep'), args.format)
    else:
        print(f'[skip] {sweep} not found -- run aggregate.py --sweep first')
    gc = args.indir / 'grad_conflict_curve.csv'
    if gc.exists():
        made += fig_grad_conflict(gc, str(args.outdir / 'fig_grad_conflict'), args.format)
    else:
        print(f'[skip] {gc} not found -- run aggregate.py --grad-conflict first')
    me = args.indir / 'model_error_curve.csv'
    if me.exists():
        made += fig_model_error(me, str(args.outdir / 'fig_model_error'), args.format)
    else:
        print(f'[skip] {me} not found -- run aggregate.py --model-error first')
    cr = args.indir / 'probe_r2_median.csv'
    if cr.exists():
        made += fig_context_recovery(cr, str(args.outdir / 'fig_context_recovery'), args.format)
    else:
        print(f'[skip] {cr} not found -- NO PROBE DATA yet '
              '(launch eval_carl_probe_taskid_supervised.sh, then run probe_recovery.py)')
    print('Wrote:')
    for f in made:
        print(' ', f)


if __name__ == '__main__':
    main()

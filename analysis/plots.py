"""
Publication figures for the CARL generalization comparison (TODO #4), built from
the aggregate.py outputs (analysis/out/*.csv).

Two figures are ready from the data that exists now:
  A. IQM learning curves, task_id vs supervised, with 95% stratified-bootstrap
     CI bands (from iqm_curves.csv).
  B. Per-task return at the final checkpoint, task_id vs supervised, sorted by
     the supervised-minus-task_id gap (from paired_per_task.csv) -- the
     redistribution result: supervised lifts the backward-walker tasks while
     ceding the forward ones.

Design follows the dataviz skill: categorical hues assigned by ENTITY and held
stable across figures (blue=task_id, aqua=supervised), thin marks, recessive
axes/grid, text in ink tokens (never the series color), a legend plus direct
labels (aqua is sub-3:1 on the light surface, so the relief rule requires
visible labels -- and the CSVs are the table view).

Not built yet (need more data): single-task-expert overlay / negative-transfer
gap (needs per-task CARL expert returns), sweep-magnitude and model-error cuts
(separate pulls), and the context-recovery scatter (needs the probe, TODO #3).
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
    print('Wrote:')
    for f in made:
        print(' ', f)


if __name__ == '__main__':
    main()

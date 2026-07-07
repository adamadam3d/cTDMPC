import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
import warnings
warnings.filterwarnings('ignore')

from glob import glob
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.logger import Logger
from common import math as tdmath
from tdmpc2 import TDMPC2

# CARL imports
from carl.envs import CARLDmcWalkerEnv, CARLDmcFishEnv, CARLDmcFingerEnv, CARLDmcQuadrupedEnv

# Mapping domains to their corresponding CARL environments
CARL_ENV_MAP = {
    'walker': CARLDmcWalkerEnv,
    'fish': CARLDmcFishEnv,
    'finger': CARLDmcFingerEnv,
    'quadruped': CARLDmcQuadrupedEnv
}

# Every dm_control task CARL can wrap for these 4 domains, including the tasks beyond
# the mt30 subset (e.g. walker-arabesque). Quadruped is excluded here; it only runs
# with -g. Only runs with -F.
FULL_CARL_TASKS = [
    'walker-walk', 'walker-run', 'walker-walk-backwards', 'walker-run-backwards',
    'walker-arabesque', 'walker-lie-down', 'walker-legs-up', 'walker-headstand', 'walker-flip', 'walker-backflip',
    'fish-upright', 'fish-swim', 'fish-obstacles',
    'finger-spin', 'finger-turn-easy', 'finger-turn-hard',
]

# Quadruped tasks. Excluded from the default and FULL (-F) task lists; only runs
# with -g.
QUADRUPED_TASKS = ['quadruped-walk', 'quadruped-run', 'quadruped-escape', 'quadruped-fetch']

from envs.dmcontrol import suite
from dm_control.rl.control import PhysicsError
import gymnasium as gym
import numpy as np

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

class CARL_TDMPC2_Wrapper(gym.Wrapper):
    def __init__(self, env, action_repeat=2):
        super().__init__(env)
        self.action_repeat = action_repeat
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=self.env.action_space.shape, dtype=np.float32)
        
        obs = self.reset()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)
        
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            obs = obs[0]
        if isinstance(obs, dict):
            flat_obs = []
            for k, v in obs.items():
                if k == 'context' or isinstance(v, (dict, str)):
                    continue
                flat_obs.append(np.asarray(v, dtype=np.float32).flatten())
            obs = np.concatenate(flat_obs, dtype=np.float32)
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        low, high = self.env.action_space.low, self.env.action_space.high
        scaled_action = low + (action + 1.0) * 0.5 * (high - low)
        scaled_action = np.clip(scaled_action, low, high)
        
        reward = 0.0
        for _ in range(self.action_repeat):
            out = self.env.step(scaled_action)
            if len(out) == 5:
                obs, r, term, trunc, info = out
                done = term or trunc
            else:
                obs, r, done, info = out
            reward += r
            if done:
                break
                
        if isinstance(obs, dict):
            flat_obs = []
            for k, v in obs.items():
                if k == 'context' or isinstance(v, (dict, str)):
                    continue
                flat_obs.append(np.asarray(v, dtype=np.float32).flatten())
            obs = np.concatenate(flat_obs, dtype=np.float32)
        return np.array(obs, dtype=np.float32), reward, done, info

from envs.wrappers.timeout import Timeout
from envs.wrappers.tensor import TensorWrapper
from envs import make_env as make_original_env

def make_carl_env(cfg, domain, task, contexts=None):
    if (domain, task) not in suite.ALL_TASKS:
        raise ValueError('Unknown task:', task)
    
    # Initialize CARL DMC environment
    carl_env_cls = CARL_ENV_MAP.get(domain)
    if not carl_env_cls:
        raise ValueError(f'CARL does not support domain: {domain} in this script')
        
    env = carl_env_cls(
        task=task,
        contexts=contexts,
        hide_context=True,
        task_kwargs={'random': cfg.seed},
        visualize_reward=False
    )
    
    # Apply standard TD-MPC2 wrappers
    env = CARL_TDMPC2_Wrapper(env, action_repeat=2)
    env = Timeout(env, max_episode_steps=500)
    env = TensorWrapper(env)

    return env


def compute_physical_smax(default_context, context_space, cap=0.95):
    """
    Largest symmetric magnitude s such that scaling EVERY numeric context feature by
    [1-s, 1+s] simultaneously stays within CARL's own declared feasible bounds for
    this domain/task (via context_space.get_lower_and_upper_bound). Features with an
    infinite bound on a side don't constrain that side; features with a zero default
    can't be constrained multiplicatively and are skipped. `cap` keeps the result
    strictly inside the boundary (rather than exactly on it) to avoid degenerate
    physics (e.g. exactly zero friction).
    """
    s_max = cap
    for name, default_value in default_context.items():
        if not isinstance(default_value, (int, float)) or 'timestep' in name.lower():
            continue
        if default_value == 0:
            continue
        try:
            lo, hi = context_space.get_lower_and_upper_bound(name)
        except Exception:
            continue
        if np.isfinite(lo):
            s_max = min(s_max, 1.0 - lo / default_value)
        if np.isfinite(hi):
            s_max = min(s_max, hi / default_value - 1.0)
    return max(s_max, 0.0)


# Reference magnitudes for context features whose default is 0.0, where multiplicative
# scaling would be a no-op (0 * mult == 0 for any mult). Perturbation for these is
# additive instead: value = default + reference_scale * (mult - 1.0), clipped to
# CARL's declared bounds. Chosen as a physically plausible "typical" magnitude for
# each quantity (water-like density/viscosity, moderate wind speed, a joint stiffness
# comparable in scale to the default joint_damping of 1.0).
REFERENCE_SCALES = {
    'density': 1000.0,
    'viscosity': 0.01,
    'joint_stiffness': 1.0,
    'wind_x': 5.0,
    'wind_y': 5.0,
    'wind_z': 5.0,
}

# Max attempts to resample a random context at a given magnitude before giving up on
# a scenario due to CARL raising a physics-infeasibility error (e.g. finger geometry
# where limb lengths can't jointly reach the spinner -- a joint constraint that isn't
# captured by any single feature's own [lo, hi] bound).
MAX_RETRY_ATTEMPTS = 15


def perturb_value(name, default_value, mult, context_space):
    """
    Perturb a single context feature by `mult` (a multiplier around 1.0). Nonzero
    defaults are scaled multiplicatively (value = default * mult). Zero defaults
    (density, viscosity, joint_stiffness, wind_*) use an additive perturbation via
    REFERENCE_SCALES instead, since 0 * mult is always 0; the result is clipped to
    the feature's declared bounds.
    """
    if default_value != 0:
        return default_value * mult
    ref = REFERENCE_SCALES.get(name)
    if ref is None:
        return default_value
    value = default_value + ref * (mult - 1.0)
    try:
        lo, hi = context_space.get_lower_and_upper_bound(name)
        if np.isfinite(lo):
            value = max(value, lo)
        if np.isfinite(hi):
            value = min(value, hi)
    except Exception:
        pass
    return value


def build_perturbed_context(default_context, s, rng, context_space):
    """Sample a fresh context at sweep magnitude s: every numeric, non-timestep
    feature perturbed independently via perturb_value with mult ~ U(1-s, 1+s)."""
    ctx = default_context.copy()
    for feature_name, default_value in default_context.items():
        if not isinstance(default_value, (int, float)) or 'timestep' in feature_name.lower():
            continue
        mult = rng.uniform(1.0 - s, 1.0 + s)
        ctx[feature_name] = perturb_value(feature_name, default_value, mult, context_space)
    return ctx


def find_checkpoints(checkpoint):
    """
    Expand `checkpoint` into an ordered list of checkpoint files. Accepts a
    single .pt file, a directory of checkpoints (e.g. the `models/` dir written
    during training), or a glob pattern. Ordered by the training iteration in
    the filename, with non-numeric names (e.g. `final.pt`) last. Mirrors
    evaluate_checkpoints.py.
    """
    path = Path(checkpoint)
    if path.is_file():
        fps = [path]
    elif path.is_dir():
        fps = [Path(fp) for fp in glob(str(path / '*.pt'))]
    else:
        fps = [Path(fp) for fp in glob(str(checkpoint))]
    assert len(fps) > 0, f'No checkpoints found at {checkpoint}'
    return sorted(fps, key=lambda fp: (0, int(fp.stem), '') if fp.stem.isdigit() else (1, 0, fp.stem))


@hydra.main(config_name='config', config_path='.')
def evaluate_carl(cfg: dict):
    assert torch.cuda.is_available(), "CUDA required for TD-MPC2."
    assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
    cfg = parse_cfg(cfg)
    
    # Use explicitly passed --seed if available, otherwise use cfg.seed
    eval_seed = os.environ.get('EVAL_SEED')
    if eval_seed is not None:
        cfg.seed = int(eval_seed)
        
    set_seed(cfg.seed)

    # --sweep : magnitude sweep over ALL context params at once, scaled within
    # [1-s, 1+s] per value of s (s=0 is always included as the unperturbed baseline).
    # Two forms:
    #   --sweep 0.1,0.2,0.3      explicit list of magnitudes
    #   --sweep auto[:n_steps]   auto: sweep baseline -> the physically-extreme
    #                            magnitude allowed by CARL's own declared context
    #                            bounds for each task, in n_steps (default 5)
    SWEEP_SCALES = os.environ.get('SWEEP_SCALES')
    sweep_auto = False
    sweep_n_steps = 5
    sweep_values = None
    if SWEEP_SCALES:
        if SWEEP_SCALES.lower().split(':')[0] in ('auto', 'max', 'extreme'):
            sweep_auto = True
            if ':' in SWEEP_SCALES:
                sweep_n_steps = int(SWEEP_SCALES.split(':', 1)[1])
            print(colored(f'Sweep mode enabled (AUTO): {sweep_n_steps} steps from baseline (s=0) to the '
                          f'physically-extreme magnitude allowed by CARL context bounds, per task.', 'yellow', attrs=['bold']))
        else:
            sweep_values = sorted(set([0.0] + [float(x) for x in SWEEP_SCALES.split(',')]))
            print(colored(f'Sweep mode enabled: magnitudes s={sweep_values} '
                          f'(all context params scaled within [1-s, 1+s] simultaneously)', 'yellow', attrs=['bold']))
    sweep_enabled = sweep_auto or sweep_values is not None

    # --probe : RQ1 context-recovery dataset. Replaces scenario evaluation entirely:
    # for each task, roll out N episodes on the FINAL checkpoint, each under a fresh
    # context with every numeric feature perturbed independently within the task's
    # physically-feasible box, and dump one row per episode pairing the inferred
    # z_ctx with the ground-truth context values (see probe_one_checkpoint).
    PROBE_EPISODES = os.environ.get('PROBE_EPISODES')
    probe_enabled = PROBE_EPISODES is not None
    if probe_enabled:
        assert getattr(cfg, 'multitask', False), \
            '--probe requires a multitask context encoder (z_ctx is None in single-task mode).'
        print(colored(f'Probe mode enabled: {PROBE_EPISODES} episodes per task, every context feature '
                      f'perturbed independently; scenario evaluation is skipped.', 'yellow', attrs=['bold']))
        if sweep_enabled:
            print(colored('--sweep is ignored in probe mode.', 'red'))

    BIGPICTURE = os.environ.get('BIGPICTURE') == '1' or cfg.get('BIGPICTURE', False) or cfg.get('bigpicture', False)
    FULL_CARL = os.environ.get('FULL_CARL') == '1'
    QUADRUPED = os.environ.get('QUADRUPED') == '1'

    eval_tasks_override = os.environ.get('EVAL_TASKS')
    if eval_tasks_override:
        target_tasks = eval_tasks_override.split(',')
    elif QUADRUPED:
        target_tasks = QUADRUPED_TASKS
        print(colored(f"QUADRUPED mode (-g) enabled: running {QUADRUPED_TASKS}", "yellow", attrs=['bold']))
    elif FULL_CARL:
        if getattr(cfg, 'multitask', False):
            # Restrict to tasks actually in this checkpoint's training set (e.g. mt30),
            # excluding CARL-wrappable tasks the model never saw (walker-arabesque, etc.).
            target_tasks = [t for t in FULL_CARL_TASKS if t in cfg.tasks]
            excluded = [t for t in FULL_CARL_TASKS if t not in cfg.tasks]
            print(colored(f"FULL mode (-F) enabled: running {len(target_tasks)} CARL-wrappable "
                          f"dm_control tasks in {cfg.task}'s training set: {target_tasks}",
                          "yellow", attrs=['bold']))
            if excluded:
                print(colored(f"  Excluded (not in {cfg.task}): {excluded}", "red"))
        else:
            target_tasks = FULL_CARL_TASKS
            print(colored(f"FULL mode (-F) enabled: running all {len(FULL_CARL_TASKS)} CARL-wrappable "
                          f"dm_control tasks across walker/fish/finger.", "yellow", attrs=['bold']))
        print(colored("Quadruped tasks are excluded from -F; pass -g to run those instead.", "red"))
    elif BIGPICTURE:
        target_tasks = ['walker-run', 'fish-swim', 'finger-spin']
        print(colored("BIGPICTURE mode (-B) enabled: running walker-run, fish-swim, finger-spin", "yellow", attrs=['bold']))
        print(colored("Note: quadruped tasks are excluded by default (pass -g to run those instead).", "red"))
        print(colored("Note: cup-spin is omitted because the CARL benchmark library does not implement a context wrapper for the cup domain.", "red"))
    else:
        # Subset of mt30 for walker, fish, and finger
        target_tasks = [
            'walker-walk', 'walker-run', 'walker-walk-backwards', 'walker-run-backwards',
            'fish-swim',
            'finger-spin', 'finger-turn-easy', 'finger-turn-hard',
        ]
    
    print(colored(f'Evaluating CARL modified environments for tasks: {target_tasks}', 'yellow', attrs=['bold']))
    
    # Load agent (using original make_env to properly initialize config for multitask, e.g., cfg.tasks)
    _ = make_original_env(cfg)
    agent = TDMPC2(cfg)

    # Expand `checkpoint` (single .pt file, directory of checkpoints, or glob
    # pattern) into an ordered list so a whole training run can be evaluated in
    # one invocation, mirroring evaluate_checkpoints.py -- including sharding
    # (`checkpoint_shard`/`num_shards`) and resume via carl_evaluated.txt.
    done_fp = None
    shard = cfg.get('checkpoint_shard', 0)
    if cfg.checkpoint != '???':
        fps = find_checkpoints(cfg.checkpoint)
        num_shards = cfg.get('num_shards', 1)
        # Optionally thin the checkpoint list for a cheaper coarse-grid backfill
        # (e.g. every-100k instead of every-50k). find_checkpoints sorts the
        # trained model (final.pt, a non-numeric stem) LAST, so fps[-1] is always
        # the fully-trained checkpoint; ALWAYS keep it -- it is the single most
        # important eval point and is not a duplicate of any numeric checkpoint
        # (offline_trainer's loop stops one step short of cfg.steps, so no
        # {cfg.steps}.pt exists; final.pt is the only copy of the converged model).
        ckpt_stride = int(cfg.get('ckpt_stride', 1))
        if ckpt_stride > 1 and not probe_enabled and len(fps) > 1:
            kept = fps[::ckpt_stride]
            if fps[-1] not in kept:
                kept.append(fps[-1])
            print(colored(f'ckpt_stride={ckpt_stride}: evaluating {len(kept)}/{len(fps)} '
                          f'checkpoints (final.pt always kept).', 'blue', attrs=['bold']))
            fps = kept
        if probe_enabled:
            # Probe the FINAL checkpoint only: RQ1 needs the converged embedding,
            # not the training trajectory. num_shards/checkpoint_shard split the
            # per-task episode budget inside probe_one_checkpoint, not the
            # checkpoint list; resume is tracked per (checkpoint, task) in
            # context_probe_done_shard<i>.txt, not carl_evaluated.txt.
            fps = fps[-1:]
            print(colored(f'Probe mode: final checkpoint only ({fps[0]}).', 'blue', attrs=['bold']))
        elif num_shards > 1:
            assert 0 <= shard < num_shards, f'checkpoint_shard must be in [0, {num_shards}), got {shard}.'
            # Stride over the full sorted list so concurrent shards are disjoint by
            # construction, regardless of which checkpoints are already evaluated.
            fps = fps[shard::num_shards]
            print(colored(f'Shard {shard}/{num_shards}: {len(fps)} checkpoint(s).', 'blue', attrs=['bold']))
        if not probe_enabled:
            print(colored(f'Found {len(fps)} checkpoint(s):', 'blue', attrs=['bold']))
            for fp in fps:
                print(colored(f'  {fp}', 'blue'))
            # Skip checkpoints already evaluated by a previous invocation, so a crashed
            # or extended sweep does not redo finished work.
            done_fp = Path(cfg.work_dir) / 'carl_evaluated.txt'
            done = set(done_fp.read_text().split()) if done_fp.exists() else set()
            if done:
                skipped = [fp for fp in fps if fp.stem in done]
                if skipped:
                    print(colored(f'Skipping {len(skipped)} checkpoint(s) already evaluated '
                                  f'(delete {done_fp} to force re-evaluation).', 'yellow', attrs=['bold']))
                    fps = [fp for fp in fps if fp.stem not in done]
            if not fps:
                print(colored('All checkpoints have already been evaluated.', 'yellow', attrs=['bold']))
                return
    else:
        print(colored('Warning: No checkpoint provided. Evaluating untrained agent.', 'red', attrs=['bold']))
        fps = [None]

    logger = Logger(cfg)
    csv_fp = Path(cfg.work_dir) / f'carl_metrics_shard{shard}.csv'

    for ckpt_fp in fps:
        step = 0
        if ckpt_fp is not None:
            step = agent.load(ckpt_fp) or 0
            if not step and ckpt_fp.stem.isdigit():
                step = int(ckpt_fp.stem)  # older checkpoints do not store their iteration
            print(colored(f'\n===== Checkpoint {ckpt_fp} (iteration {step}) =====', 'blue', attrs=['bold']))

        # Reseed per checkpoint so every checkpoint sees the exact same sequence of
        # perturbation draws -- curves over iterations are then comparable, not
        # confounded by different random contexts per checkpoint.
        set_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)

        if probe_enabled:
            probe_one_checkpoint(cfg, agent, step, target_tasks, shard,
                                 cfg.get('num_shards', 1), int(PROBE_EPISODES))
            continue

        metrics = {'iteration': step}
        baseline_scores = []  # evaluate.py-style normalized score, from each task's unperturbed baseline
        all_retentions = []   # retention (perturbed/baseline) from every non-baseline scenario, all tasks

        eval_one_checkpoint(cfg, agent, logger, target_tasks, sweep_enabled, sweep_auto,
                            sweep_n_steps, sweep_values, BIGPICTURE, rng, metrics,
                            baseline_scores, all_retentions)

        if baseline_scores:
            # nanmean: a single task with an undefined (nan) score (e.g. zero
            # baseline reward) must not blank out the whole checkpoint's summary.
            metrics['baseline_normalized_score'] = np.nanmean(baseline_scores)
            print(colored(f'\nBaseline Normalized Score (s=0 / unperturbed, mt30-style): {metrics["baseline_normalized_score"]:.02f}', 'yellow', attrs=['bold']))
        if all_retentions:
            n_nan = np.sum(np.isnan(all_retentions))
            metrics['overall_mean_retention'] = np.nanmean(all_retentions)
            nan_note = f' ({n_nan}/{len(all_retentions)} scenarios had undefined retention: zero baseline reward)' if n_nan else ''
            print(colored(f'Overall Mean Retention across perturbed scenarios (all tasks): {metrics["overall_mean_retention"]:.02f}{nan_note}', 'yellow', attrs=['bold']))
        if logger.wandb:
            # Same convention as evaluate_checkpoints.py: `iteration` is the x-axis,
            # metrics are prefixed with the category by Logger.log.
            logger.log(metrics, 'pretrain')
        # Local, wandb-independent record of all metrics, one CSV per shard;
        # append so resumed invocations keep earlier rows.
        pd.DataFrame([metrics]).to_csv(csv_fp, mode='a', header=not csv_fp.exists(), index=False)
        if done_fp is not None:
            with open(done_fp, 'a') as f:
                f.write(f'{ckpt_fp.stem}\n')

    logger.finish()


def eval_one_checkpoint(cfg, agent, logger, target_tasks, sweep_enabled, sweep_auto,
                        sweep_n_steps, sweep_values, BIGPICTURE, rng, metrics,
                        baseline_scores, all_retentions):
    """Run the CARL evaluation of the currently-loaded agent over `target_tasks`,
    accumulating per-scenario results into `metrics` (keyed for wandb/CSV) and the
    `baseline_scores`/`all_retentions` aggregate lists."""
    for task_str in target_tasks:
        domain, task = task_str.replace('-', '_').split('_', 1)
        domain = dict(cup='ball_in_cup', pointmass='point_mass').get(domain, domain)
        
        # Identify task index if running in multitask mode
        task_idx = None
        if getattr(cfg, 'multitask', False):
            if task_str in cfg.tasks:
                task_idx = cfg.tasks.index(task_str)
            else:
                print(colored(f'Task {task_str} not found in training tasks. Evaluating zero-shot transfer!', 'magenta'))
                if cfg.context_encoder == 'task_id':
                    print(colored(f'WARNING: task_id encoder cannot do zero-shot transfer properly. It will blindly use the embedding of the first task!', 'red'))
                # We need a task_idx for action masking in TD-MPC2. 
                # Pick the task index that has the largest action dimension to prevent masking valid unseen actions.
                task_idx = int(np.argmax(cfg.action_dims))
        
        print(colored(f'\n--- Task: {task_str} ---', 'magenta', attrs=['bold']))
        
        # Instantiate once to extract the default contexts available for this domain/task
        carl_env_cls = CARL_ENV_MAP[domain]
        temp_env = carl_env_cls(task=task)
        # Use default context from the CARL environment
        default_context = temp_env.get_default_context()
        context_space = carl_env_cls.get_context_space()
        baseline_reward = None  # set on the first (s=0) scenario when sweeping

        # Each entry: (label, ctx_or_sampler, is_baseline). ctx_or_sampler is either a
        # static context dict, or (for randomized sweep scenarios) a zero-arg callable
        # that draws a fresh random context each time it's called -- used to retry on
        # physics-infeasible draws instead of giving up on the whole scenario.
        eval_scenarios = []

        if sweep_enabled:
            if sweep_auto:
                s_max = compute_physical_smax(default_context, context_space)
                task_sweep_values = list(np.linspace(0.0, s_max, sweep_n_steps))
                print(colored(f'  [auto] physically-extreme magnitude for {task_str}: s_max={s_max:.4f}', 'cyan'))
            else:
                task_sweep_values = sweep_values

            # Dose-response sweep: one scenario per magnitude s, all numeric context
            # params perturbed simultaneously within [1-s, 1+s] (multiplicatively for
            # nonzero defaults, additively via REFERENCE_SCALES for zero defaults --
            # see perturb_value). s=0 is the unperturbed baseline (evaluated first).
            for s in task_sweep_values:
                if s == 0.0:
                    eval_scenarios.append((f'Sweep s=0.00 (Baseline)', default_context.copy(), True))
                else:
                    label = f'Sweep s={s:.2f} (scale [{1-s:.2f}, {1+s:.2f}])'
                    sampler = (lambda s=s: build_perturbed_context(default_context, s, rng, context_space))
                    eval_scenarios.append((label, sampler, False))
        else:
            RUN_RANDOM = os.environ.get('RUN_RANDOM') == '1'
            RUN_HIGH = os.environ.get('RUN_HIGH') == '1'
            RUN_LOW = os.environ.get('RUN_LOW') == '1'
            RUN_NORMAL = os.environ.get('RUN_NORMAL') == '1'
            if not (RUN_RANDOM or RUN_HIGH or RUN_LOW or RUN_NORMAL):
                RUN_RANDOM = RUN_HIGH = RUN_LOW = RUN_NORMAL = True

            if RUN_NORMAL:
                eval_scenarios.append(("Baseline / Normal (1.0x)", default_context.copy(), True))

            if BIGPICTURE:
                all_low = default_context.copy()
                all_high = default_context.copy()
                all_random = default_context.copy()
                for feature_name, default_value in default_context.items():
                    if not isinstance(default_value, (int, float)) or 'timestep' in feature_name.lower():
                        continue
                    all_low[feature_name] = perturb_value(feature_name, default_value, 0.5, context_space)
                    all_high[feature_name] = perturb_value(feature_name, default_value, 1.5, context_space)
                    # Random multiplier between 0.5 and 1.5
                    all_random[feature_name] = perturb_value(feature_name, default_value, np.random.uniform(0.5, 1.5), context_space)
                if RUN_RANDOM:
                    eval_scenarios.append(("All Params Random (0.5x - 1.5x)", all_random, False))
                if RUN_LOW:
                    eval_scenarios.append(("All Params Low (-50%)", all_low, False))
                if RUN_HIGH:
                    eval_scenarios.append(("All Params High (+50%)", all_high, False))
            else:
                for feature_name, default_value in default_context.items():
                    if not isinstance(default_value, (int, float)) or 'timestep' in feature_name.lower():
                        continue
                    ctx_low = default_context.copy()
                    ctx_low[feature_name] = perturb_value(feature_name, default_value, 0.5, context_space)

                    ctx_high = default_context.copy()
                    ctx_high[feature_name] = perturb_value(feature_name, default_value, 1.5, context_space)

                    ctx_random = default_context.copy()
                    random_mult = np.random.uniform(0.5, 1.5)
                    ctx_random[feature_name] = perturb_value(feature_name, default_value, random_mult, context_space)

                    if RUN_RANDOM:
                        eval_scenarios.append((f"{feature_name} = {ctx_random[feature_name]:.4f} (Random {random_mult:.2f}x)", ctx_random, False))
                    if RUN_LOW:
                        eval_scenarios.append((f"{feature_name} = {ctx_low[feature_name]:.4f} (Low)", ctx_low, False))
                    if RUN_HIGH:
                        eval_scenarios.append((f"{feature_name} = {ctx_high[feature_name]:.4f} (High)", ctx_high, False))

        for scn_idx, (mod_label, ctx_dict_or_sampler, is_baseline) in enumerate(eval_scenarios):
            print(colored(f'Evaluating {mod_label}', 'cyan'))

            # Retry the WHOLE attempt (context draw + env construction + full episode
            # rollout) on failure. Two distinct failure modes can occur: CARL raises
            # ValueError at construction time for a jointly-infeasible context (e.g.
            # finger geometry that can't reach the spinner), or MuJoCo raises
            # PhysicsError mid-rollout when an extreme perturbation makes the
            # simulation numerically unstable (NaN/Inf/huge QACC). Both are treated
            # the same: for randomized scenarios, resample and try again; for
            # deterministic scenarios (baseline/Low/High) there's nothing to resample,
            # so a single failure just skips that scenario.
            is_random = callable(ctx_dict_or_sampler)
            max_attempts = MAX_RETRY_ATTEMPTS if is_random else 1
            ep_rewards, ep_successes = None, None
            last_err = None

            for _attempt in range(max_attempts):
                ctx = ctx_dict_or_sampler() if is_random else ctx_dict_or_sampler
                try:
                    env = make_carl_env(cfg, domain, task, contexts={0: ctx})
                except ValueError as e:
                    last_err = e
                    continue

                try:
                    ep_rewards, ep_successes = [], []
                    ep_consistency_errs, ep_reward_errs = [], []
                    for i in range(cfg.eval_episodes):
                        obs, done, ep_reward, t = env.reset(), False, 0, 0
                        while not done:
                            # Multitask agents expect padded observations and output padded actions
                            is_mt = getattr(cfg, 'multitask', False)
                            if is_mt:
                                expected_obs_dim = max(cfg.obs_shapes)
                                if obs.shape[0] < expected_obs_dim:
                                    padding = torch.zeros(expected_obs_dim - obs.shape[0], dtype=obs.dtype, device=obs.device)
                                    padded_obs = torch.cat((obs, padding))
                                elif obs.shape[0] > expected_obs_dim:
                                    padded_obs = obs[:expected_obs_dim]
                                else:
                                    padded_obs = obs
                            else:
                                padded_obs = obs

                            # Agent act uses task_idx for encoders like task_id
                            action = agent.act(padded_obs, t0=t==0, task=task_idx)
                            prev_obs = padded_obs

                            env_action = action
                            if env_action.shape[0] < env.action_space.shape[0]:
                                padding = torch.zeros(env.action_space.shape[0] - env_action.shape[0], dtype=env_action.dtype, device=env_action.device)
                                env_action = torch.cat((env_action, padding))
                            elif env_action.shape[0] > env.action_space.shape[0]:
                                env_action = env_action[:env.action_space.shape[0]]

                            obs, reward, done, info = env.step(env_action)

                            # Multi-task context encoders require updating context
                            if is_mt:
                                if obs.shape[0] < expected_obs_dim:
                                    next_padded_obs = torch.cat((obs, torch.zeros(expected_obs_dim - obs.shape[0], dtype=obs.dtype, device=obs.device)))
                                elif obs.shape[0] > expected_obs_dim:
                                    next_padded_obs = obs[:expected_obs_dim]
                                else:
                                    next_padded_obs = obs

                                # Model-prediction diagnostics on this real transition,
                                # under z_ctx_t -- the context that was actually used to
                                # select `action` (recovered via agent.context, called
                                # before update_context mutates the online context state
                                # below). Mirrors the training-time consistency/reward
                                # losses (Eq. loss-model): does the world model, given the
                                # inferred context, predict the real next latent state and
                                # reward, on a context possibly unseen during training?
                                with torch.no_grad():
                                    z_ctx_t = agent.context(task_idx, eval_mode=False)
                                    obs_t = prev_obs.unsqueeze(0).to(agent.device)
                                    act_t = action.unsqueeze(0).to(agent.device)
                                    next_obs_t = next_padded_obs.unsqueeze(0).to(agent.device)
                                    z_t = agent.model.encode(obs_t, z_ctx_t)
                                    z_pred = agent.model.next(z_t, act_t, z_ctx_t)
                                    r_pred = tdmath.two_hot_inv(agent.model.reward(z_t, act_t, z_ctx_t), cfg)
                                    z_true = agent.model.encode(next_obs_t, z_ctx_t)
                                    ep_consistency_errs.append(F.mse_loss(z_pred, z_true).item())
                                    ep_reward_errs.append((r_pred.item() - reward) ** 2)

                                agent.update_context(prev_obs, action, reward, next_padded_obs)

                            ep_reward += reward
                            t += 1

                        ep_rewards.append(ep_reward)
                        ep_successes.append(info.get('success', 0.0))
                    break  # all episodes completed without a physics failure
                except PhysicsError as e:
                    last_err = e
                    ep_rewards, ep_successes = None, None
                    continue

            if ep_rewards is None:
                reason = f'{max_attempts} resample attempts' if is_random else 'the attempt'
                print(colored(f"  Skipping scenario after {reason} "
                              f"(physics infeasible/unstable): {last_err}", "red"))
                continue

            mean_reward = np.mean(ep_rewards)
            mean_success = np.mean(ep_successes)
            # NaN (not 0) when empty: single-task mode never populates these, and
            # a genuine "0 error" reading must stay distinguishable from "not measured".
            mean_consistency_err = np.mean(ep_consistency_errs) if ep_consistency_errs else float('nan')
            mean_reward_err = np.mean(ep_reward_errs) if ep_reward_errs else float('nan')

            if is_baseline and getattr(cfg, 'multitask', False):
                # Same normalization convention as evaluate.py's multitask score.
                baseline_scores.append(mean_success * 100 if task_str.startswith('mw-') else mean_reward / 10)

            if sweep_enabled:
                if baseline_reward is None:
                    baseline_reward = mean_reward
                retention = mean_reward / baseline_reward if baseline_reward != 0 else float('nan')
                if not is_baseline:
                    all_retentions.append(retention)
                print(colored(f'  Result -> R: {mean_reward:.01f} | S: {mean_success:.02f} | Retention: {retention:.02f}', 'green'))
            else:
                print(colored(f'  Result -> R: {mean_reward:.01f} | S: {mean_success:.02f}', 'green'))

            # Accumulate into the per-checkpoint metrics dict; the baseline scenario
            # keeps the bare task key (comparable to evaluate_checkpoints.py curves),
            # perturbed scenarios are suffixed by their stable scenario index.
            if is_baseline:
                metrics[f'episode_reward+{task_str}'] = mean_reward
                metrics[f'episode_success+{task_str}'] = mean_success
                metrics[f'consistency_error+{task_str}'] = mean_consistency_err
                metrics[f'reward_error+{task_str}'] = mean_reward_err
            else:
                metrics[f'episode_reward+{task_str}+scn{scn_idx}'] = mean_reward
                metrics[f'episode_success+{task_str}+scn{scn_idx}'] = mean_success
                metrics[f'consistency_error+{task_str}+scn{scn_idx}'] = mean_consistency_err
                metrics[f'reward_error+{task_str}+scn{scn_idx}'] = mean_reward_err
                if sweep_enabled:
                    metrics[f'retention+{task_str}+scn{scn_idx}'] = retention


# Context-recovery probe (--probe): z_ctx snapshot times within an episode. The
# online context window (cfg.context_window=100) is full from t=100 on, so the
# t<100 snapshots capture the identification transient and the tail mean is the
# settled embedding. All steps within an episode share ONE ground-truth context,
# so the probe's effective sample size is #episodes, not #steps -- hence one row
# per episode with a few snapshots rather than per-step rows.
PROBE_SNAP_STEPS = (25, 50, 100, 250)
PROBE_TAIL_STEPS = 100


def probe_rollout(cfg, agent, env, task_idx):
    """
    One full episode under the currently-loaded agent, collecting the inferred
    context embedding after every step. Mirrors the scenario rollout in
    eval_one_checkpoint (obs/action padding, model-prediction diagnostics) but
    returns the per-step z_ctx history instead of accumulating wandb metrics.
    PhysicsError propagates to the caller, which resamples the context.
    """
    expected_obs_dim = max(cfg.obs_shapes)

    def pad_obs(o):
        if o.shape[0] < expected_obs_dim:
            return torch.cat((o, torch.zeros(expected_obs_dim - o.shape[0], dtype=o.dtype, device=o.device)))
        if o.shape[0] > expected_obs_dim:
            return o[:expected_obs_dim]
        return o

    obs, done, ep_reward, t = env.reset(), False, 0, 0
    z_hist, cons_errs, rew_errs = [], [], []
    while not done:
        padded_obs = pad_obs(obs)
        action = agent.act(padded_obs, t0=t == 0, task=task_idx)
        prev_obs = padded_obs

        env_action = action
        if env_action.shape[0] < env.action_space.shape[0]:
            padding = torch.zeros(env.action_space.shape[0] - env_action.shape[0], dtype=env_action.dtype, device=env_action.device)
            env_action = torch.cat((env_action, padding))
        elif env_action.shape[0] > env.action_space.shape[0]:
            env_action = env_action[:env.action_space.shape[0]]

        obs, reward, done, info = env.step(env_action)
        next_padded_obs = pad_obs(obs)

        with torch.no_grad():
            # Same model-prediction diagnostics as eval_one_checkpoint, under the
            # z_ctx that actually selected `action` (read before update_context).
            z_ctx_t = agent.context(task_idx, eval_mode=False)
            obs_t = prev_obs.unsqueeze(0).to(agent.device)
            act_t = action.unsqueeze(0).to(agent.device)
            next_obs_t = next_padded_obs.unsqueeze(0).to(agent.device)
            z_t = agent.model.encode(obs_t, z_ctx_t)
            z_pred = agent.model.next(z_t, act_t, z_ctx_t)
            r_pred = tdmath.two_hot_inv(agent.model.reward(z_t, act_t, z_ctx_t), cfg)
            z_true = agent.model.encode(next_obs_t, z_ctx_t)
            cons_errs.append(F.mse_loss(z_pred, z_true).item())
            rew_errs.append((r_pred.item() - reward) ** 2)

        agent.update_context(prev_obs, action, reward, next_padded_obs)
        with torch.no_grad():
            # The embedding after t+1 observed transitions: the probe's regressor input.
            z_hist.append(agent.context(task_idx, eval_mode=False).squeeze(0).float().cpu().numpy())

        ep_reward += reward
        t += 1

    return ep_reward, info.get('success', 0.0), cons_errs, rew_errs, z_hist


def probe_one_checkpoint(cfg, agent, step, target_tasks, shard, num_shards, n_episodes):
    """
    Dump the RQ1 context-recovery dataset for the currently-loaded checkpoint.
    For each task, roll out this shard's slice of the n_episodes budget, each
    episode under a fresh context with EVERY numeric feature perturbed
    independently (mult ~ U(1-s_max, 1+s_max), s_max = the task's physically-
    feasible bound via compute_physical_smax). Independent per-feature draws --
    unlike the lockstep --sweep -- keep the probe's design matrix well-conditioned
    so per-dimension R^2 is attributable to that dimension.

    One row per episode: ground-truth context values (raw, unnormalized) +
    z_ctx snapshots at PROBE_SNAP_STEPS and the last-PROBE_TAIL_STEPS mean,
    appended to a per-task, per-shard CSV (per-task because context features
    differ across domains; a shared CSV would misalign columns on append).

    For task_id checkpoints the z columns are constant per task by construction:
    that is the negative control -- probe R^2 ~ 0 expected.
    """
    done_fp = Path(cfg.work_dir) / f'context_probe_done_shard{shard}.txt'
    done = set(done_fp.read_text().splitlines()) if done_fp.exists() else set()
    ep_ids = list(range(shard, n_episodes, num_shards))
    for task_str in target_tasks:
        done_key = f'{step}:{task_str}'
        if done_key in done:
            print(colored(f'Probe: skipping {task_str} (already done, see {done_fp}).', 'yellow'))
            continue
        domain, task = task_str.replace('-', '_').split('_', 1)
        domain = dict(cup='ball_in_cup', pointmass='point_mass').get(domain, domain)
        task_idx = cfg.tasks.index(task_str) if task_str in cfg.tasks else int(np.argmax(cfg.action_dims))

        carl_env_cls = CARL_ENV_MAP[domain]
        temp_env = carl_env_cls(task=task)
        default_context = temp_env.get_default_context()
        context_space = carl_env_cls.get_context_space()
        s_max = compute_physical_smax(default_context, context_space)
        # Deterministic per (seed, shard, task) stream: shards draw disjoint
        # context sequences, and a resubmission redraws the same ones.
        rng = np.random.default_rng([cfg.seed, shard, target_tasks.index(task_str)])
        print(colored(f'\n--- Probe task: {task_str} ({len(ep_ids)} episodes, s_max={s_max:.4f}) ---',
                      'magenta', attrs=['bold']))

        rows = []
        for ep_id in ep_ids:
            result, last_err = None, None
            for _attempt in range(MAX_RETRY_ATTEMPTS):
                ctx = build_perturbed_context(default_context, s_max, rng, context_space)
                try:
                    env = make_carl_env(cfg, domain, task, contexts={0: ctx})
                except ValueError as e:
                    last_err = e
                    continue
                try:
                    result = probe_rollout(cfg, agent, env, task_idx)
                    break
                except PhysicsError as e:
                    last_err = e
                    continue
            if result is None:
                print(colored(f'  Episode {ep_id}: skipped after {MAX_RETRY_ATTEMPTS} resample attempts '
                              f'(physics infeasible/unstable): {last_err}', 'red'))
                continue
            ep_reward, ep_success, cons_errs, rew_errs, z_hist = result
            row = {'iteration': step, 'task': task_str, 'episode': ep_id, 's_max': s_max,
                   'episode_reward': float(ep_reward), 'episode_success': float(ep_success),
                   'consistency_error': float(np.mean(cons_errs)) if cons_errs else float('nan'),
                   'reward_error': float(np.mean(rew_errs)) if rew_errs else float('nan')}
            for name, value in ctx.items():
                if isinstance(value, (int, float)):
                    row[f'ctx_{name}'] = value
            z_dim = len(z_hist[0])
            for snap_t in PROBE_SNAP_STEPS:
                z = z_hist[snap_t - 1] if len(z_hist) >= snap_t else [float('nan')] * z_dim
                for i in range(z_dim):
                    row[f'z_t{snap_t}_{i}'] = z[i]
            z_tail = np.mean(z_hist[-PROBE_TAIL_STEPS:], axis=0)
            for i in range(z_dim):
                row[f'z_tail_{i}'] = z_tail[i]
            rows.append(row)
            print(colored(f'  Episode {ep_id}: R: {float(ep_reward):.01f} | S: {float(ep_success):.02f}', 'green'))

        # Buffer the whole task slice and append once: a crash mid-task leaves no
        # partial rows behind, so the all-or-nothing resume marker stays accurate.
        csv_fp = Path(cfg.work_dir) / f'context_probe_{task_str}_shard{shard}.csv'
        if rows:
            pd.DataFrame(rows).to_csv(csv_fp, mode='a', header=not csv_fp.exists(), index=False)
        with open(done_fp, 'a') as f:
            f.write(f'{done_key}\n')
        print(colored(f'  Wrote {len(rows)} rows -> {csv_fp}', 'blue'))


import sys
if __name__ == '__main__':
    if '--seed' in sys.argv:
        idx = sys.argv.index('--seed')
        os.environ['EVAL_SEED'] = sys.argv[idx + 1]
        sys.argv.pop(idx)
        sys.argv.pop(idx)
    if '-B' in sys.argv:
        os.environ['BIGPICTURE'] = '1'
        sys.argv.remove('-B')
    if '-F' in sys.argv:
        os.environ['FULL_CARL'] = '1'
        sys.argv.remove('-F')
    if '-g' in sys.argv:
        os.environ['QUADRUPED'] = '1'
        sys.argv.remove('-g')
    if '-n' in sys.argv:
        os.environ['RUN_NORMAL'] = '1'
        sys.argv.remove('-n')
    if '-r' in sys.argv:
        os.environ['RUN_RANDOM'] = '1'
        sys.argv.remove('-r')
    if '-H' in sys.argv:
        os.environ['RUN_HIGH'] = '1'
        sys.argv.remove('-H')
    if '-L' in sys.argv:
        os.environ['RUN_LOW'] = '1'
        sys.argv.remove('-L')
    if '--eval_tasks' in sys.argv:
        idx = sys.argv.index('--eval_tasks')
        os.environ['EVAL_TASKS'] = sys.argv[idx + 1]
        sys.argv.pop(idx)
        sys.argv.pop(idx)
    # --probe [N] : context-recovery probe (RQ1), N episodes per task (default 200).
    # Replaces scenario evaluation; see probe_one_checkpoint.
    if '--probe' in sys.argv:
        idx = sys.argv.index('--probe')
        nxt = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else ''
        has_value = nxt != '' and '=' not in nxt and not nxt.startswith('-')
        if has_value:
            try:
                int(nxt)
            except ValueError:
                print(f"Error: --probe requires an integer episode count, got: {nxt!r}")
                sys.exit(1)
            os.environ['PROBE_EPISODES'] = nxt
            sys.argv.pop(idx)
            sys.argv.pop(idx)
        else:
            os.environ['PROBE_EPISODES'] = '200'
            sys.argv.pop(idx)
    # --sweep [auto[:n_steps] | s1,s2,...]
    # No value -> defaults to 'auto:5': sweep baseline (s=0) to the physically-extreme
    # magnitude allowed by CARL's own declared context bounds, in 5 steps.
    DEFAULT_SWEEP = 'auto:5'
    if '--sweep' in sys.argv:
        idx = sys.argv.index('--sweep')
        has_value = idx + 1 < len(sys.argv) and '=' not in sys.argv[idx + 1]
        if has_value:
            sweep_arg = sys.argv[idx + 1]
            keyword = sweep_arg.lower().split(':')[0]
            if keyword in ('auto', 'max', 'extreme'):
                if ':' in sweep_arg:
                    try:
                        int(sweep_arg.split(':', 1)[1])
                    except ValueError:
                        print(f"Error: --sweep {keyword}:<n> requires an integer step count, got: {sweep_arg!r}")
                        sys.exit(1)
            else:
                try:
                    [float(x) for x in sweep_arg.split(',')]
                except ValueError:
                    print(f"Error: --sweep values must be numbers or 'auto[:n_steps]', got: {sweep_arg!r}")
                    sys.exit(1)
            sys.argv.pop(idx)
            sys.argv.pop(idx)
        else:
            sweep_arg = DEFAULT_SWEEP
            print(f"No value given for --sweep; defaulting to {DEFAULT_SWEEP} "
                  f"(baseline -> physically-extreme CARL bound, 5 steps)")
            sys.argv.pop(idx)
        os.environ['SWEEP_SCALES'] = sweep_arg
    evaluate_carl()

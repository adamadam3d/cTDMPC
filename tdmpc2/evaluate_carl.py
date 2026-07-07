import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
import warnings
warnings.filterwarnings('ignore')

import hydra
import numpy as np
import torch
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
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
    'walker-stand', 'walker-walk', 'walker-run', 'walker-walk-backwards', 'walker-run-backwards',
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
    rng = np.random.default_rng(cfg.seed)

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
            'walker-stand', 'walker-walk', 'walker-run', 'walker-walk-backwards', 'walker-run-backwards',
            'fish-swim',
            'finger-spin', 'finger-turn-easy', 'finger-turn-hard',
        ]
    
    print(colored(f'Evaluating CARL modified environments for tasks: {target_tasks}', 'yellow', attrs=['bold']))
    
    # Load agent (using original make_env to properly initialize config for multitask, e.g., cfg.tasks)
    _ = make_original_env(cfg)
    agent = TDMPC2(cfg)
    if cfg.checkpoint != '???':
        assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found!'
        agent.load(cfg.checkpoint)
        print(colored(f'Loaded Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
    else:
        print(colored('Warning: No checkpoint provided. Evaluating untrained agent.', 'red', attrs=['bold']))

    baseline_scores = []  # evaluate.py-style normalized score, from each task's unperturbed baseline
    all_retentions = []   # retention (perturbed/baseline) from every non-baseline scenario, all tasks

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

        for mod_label, ctx_dict_or_sampler, is_baseline in eval_scenarios:
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

    if baseline_scores:
        print(colored(f'\nBaseline Normalized Score (s=0 / unperturbed, mt30-style): {np.mean(baseline_scores):.02f}', 'yellow', attrs=['bold']))
    if all_retentions:
        print(colored(f'Overall Mean Retention across perturbed scenarios (all tasks): {np.mean(all_retentions):.02f}', 'yellow', attrs=['bold']))

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

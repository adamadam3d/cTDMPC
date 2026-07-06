import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
import warnings
warnings.filterwarnings('ignore')

import hydra
import imageio
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
    'quadruped': CARLDmcQuadrupedEnv,
}

from envs.dmcontrol import suite
import gymnasium as gym

from envs.wrappers.timeout import Timeout
from envs.wrappers.tensor import TensorWrapper
from envs import make_env as make_original_env

torch.backends.cudnn.benchmark = True


class CARL_TDMPC2_Wrapper(gym.Wrapper):
    """Flattens CARL dict observations and rescales actions to [-1, 1] for TD-MPC2."""

    def __init__(self, env, action_repeat=2):
        super().__init__(env)
        self.action_repeat = action_repeat
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=self.env.action_space.shape, dtype=np.float32)

        obs = self.reset()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)

    def _flatten_obs(self, obs):
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

    def reset(self, **kwargs):
        return self._flatten_obs(self.env.reset(**kwargs))

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

        return self._flatten_obs(obs), reward, done, info


def make_carl_env(cfg, domain, task, contexts=None):
    """Build a CARL DMC environment with the given context and TD-MPC2 wrappers."""
    if (domain, task) not in suite.ALL_TASKS:
        raise ValueError('Unknown task:', task)

    carl_env_cls = CARL_ENV_MAP.get(domain)
    if not carl_env_cls:
        raise ValueError(f'CARL does not support domain: {domain} in this script')

    env = carl_env_cls(
        task=task,
        contexts=contexts,
        hide_context=True,
        task_kwargs={'random': cfg.seed},
        visualize_reward=False,
    )

    env = CARL_TDMPC2_Wrapper(env, action_repeat=2)
    env = Timeout(env, max_episode_steps=500)
    env = TensorWrapper(env)
    return env


def random_context(default_context, rng, low=0.5, high=1.5):
    """Randomly perturb every numeric context feature by a multiplier in [low, high]."""
    ctx = default_context.copy()
    for feature_name, default_value in default_context.items():
        if not isinstance(default_value, (int, float)) or 'timestep' in feature_name.lower():
            continue
        ctx[feature_name] = default_value * rng.uniform(low, high)
    return ctx


def _parse_domain_task(task_str):
    domain, task = task_str.replace('-', '_').split('_', 1)
    domain = dict(cup='ball_in_cup', pointmass='point_mass').get(domain, domain)
    return domain, task


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
    """
    Script for evaluating a single-task / multi-task TD-MPC2 checkpoint on CARL
    environments whose context (task) parameters are randomized with a fixed seed.

    Mirrors evaluate.py, but every numeric CARL context feature is multiplied by a
    random factor in [0.5, 1.5], drawn from a seeded RNG so runs are reproducible.

    Most relevant args:
        `task`: task name (or mt30/mt80 for multi-task evaluation)
        `model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
        `checkpoint`: path to model checkpoint to load
        `eval_episodes`: number of episodes to evaluate on per task (default: 10)
        `save_video`: whether to save a video of the evaluation (default: True)
        `seed`: random seed, also seeds the context randomization (default: 1)

    Example usage:
    ````
        $ python evaluate_carl_random.py task=walker-run checkpoint=/path/to/walker.pt
        $ python evaluate_carl_random.py task=mt30 model_size=317 checkpoint=/path/to/mt30-317M.pt
    ```
    """
    assert torch.cuda.is_available()
    assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
    print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
    print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
    if not cfg.multitask and ('mt80' in cfg.checkpoint or 'mt30' in cfg.checkpoint):
        print(colored('Warning: single-task evaluation of multi-task models is not currently supported.', 'red', attrs=['bold']))
        print(colored('To evaluate a multi-task model, use task=mt80 or task=mt30.', 'red', attrs=['bold']))

    # Initialize config (obs/action dims, cfg.tasks for multitask) via the original env factory.
    make_original_env(cfg)

    # Load agent
    agent = TDMPC2(cfg)
    assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
    agent.load(cfg.checkpoint)

    # Evaluate
    if cfg.multitask:
        print(colored(f'Evaluating agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
    else:
        print(colored(f'Evaluating agent on {cfg.task}:', 'yellow', attrs=['bold']))
    if cfg.save_video:
        video_dir = os.path.join(cfg.work_dir, 'videos')
        os.makedirs(video_dir, exist_ok=True)
    scores = []
    tasks = cfg.tasks if cfg.multitask else [cfg.task]
    for task_idx, task in enumerate(tasks):
        if not cfg.multitask:
            task_idx = None

        domain, dmc_task = _parse_domain_task(task)
        if domain not in CARL_ENV_MAP:
            print(colored(f'  {task:<22}\tSkipped: CARL has no context wrapper for domain "{domain}"', 'red'))
            continue

        # Sample a random context for this task from the default context.
        default_context = CARL_ENV_MAP[domain](task=dmc_task).get_default_context()
        contexts = {0: random_context(default_context, rng)}

        try:
            env = make_carl_env(cfg, domain, dmc_task, contexts=contexts)
        except ValueError as e:
            print(colored(f'  {task:<22}\tSkipped due to physics constraints: {e}', 'red'))
            continue

        # Multi-task models expect observations/actions padded to the global (max)
        # dimensions used during training; single-task envs emit smaller vectors.
        is_mt = bool(cfg.multitask)
        expected_obs_dim = max(cfg.obs_shapes) if is_mt else None
        env_action_dim = env.action_space.shape[0]

        def pad_obs(o):
            if not is_mt or o.shape[0] == expected_obs_dim:
                return o
            if o.shape[0] < expected_obs_dim:
                pad = torch.zeros(expected_obs_dim - o.shape[0], dtype=o.dtype, device=o.device)
                return torch.cat((o, pad))
            return o[:expected_obs_dim]

        def fit_action(a):
            if a.shape[0] == env_action_dim:
                return a
            if a.shape[0] < env_action_dim:
                pad = torch.zeros(env_action_dim - a.shape[0], dtype=a.dtype, device=a.device)
                return torch.cat((a, pad))
            return a[:env_action_dim]

        ep_rewards, ep_successes = [], []
        for i in range(cfg.eval_episodes):
            obs, done, ep_reward, t = pad_obs(env.reset()), False, 0, 0
            if cfg.save_video:
                frames = [env.render()]
            while not done:
                action = agent.act(obs, t0=t == 0, task=task_idx)
                prev_obs = obs
                obs, reward, done, info = env.step(fit_action(action))
                obs = pad_obs(obs)
                if cfg.multitask:
                    agent.update_context(prev_obs, action, reward, obs)
                ep_reward += reward
                t += 1
                if cfg.save_video:
                    frames.append(env.render())
            ep_rewards.append(ep_reward)
            ep_successes.append(info.get('success', 0.0))
            if cfg.save_video:
                imageio.mimsave(
                    os.path.join(video_dir, f'{task}-{i}.mp4'), frames, fps=15)
        ep_rewards = np.mean(ep_rewards)
        ep_successes = np.mean(ep_successes)
        if cfg.multitask:
            scores.append(ep_successes * 100 if task.startswith('mw-') else ep_rewards / 10)
        print(colored(f'  {task:<22}'
                      f'\tR: {ep_rewards:.01f}  '
                      f'\tS: {ep_successes:.02f}', 'yellow'))
    if cfg.multitask:
        print(colored(f'Normalized score: {np.mean(scores):.02f}', 'yellow', attrs=['bold']))


if __name__ == '__main__':
    evaluate()

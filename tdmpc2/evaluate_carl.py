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

from envs.dmcontrol import suite
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
    
    BIGPICTURE = os.environ.get('BIGPICTURE') == '1' or cfg.get('BIGPICTURE', False) or cfg.get('bigpicture', False)
    
    eval_tasks_override = os.environ.get('EVAL_TASKS')
    if eval_tasks_override:
        target_tasks = eval_tasks_override.split(',')
    elif BIGPICTURE:
        target_tasks = ['walker-run', 'fish-swim', 'finger-spin', 'quadruped-walk']
        print(colored("BIGPICTURE mode (-B) enabled: running walker-run, fish-swim, finger-spin, quadruped-walk", "yellow", attrs=['bold']))
        print(colored("Note: cup-spin is omitted because the CARL benchmark library does not implement a context wrapper for the cup domain.", "red"))
    else:
        # Subset of mt30 for walker, fish, and finger
        target_tasks = [
            'walker-stand', 'walker-walk', 'walker-run', 'walker-walk-backwards', 'walker-run-backwards',
            'fish-swim',
            'finger-spin', 'finger-turn-easy', 'finger-turn-hard',
            'quadruped-walk', 'quadruped-run'
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
        
        eval_scenarios = []
        RUN_RANDOM = os.environ.get('RUN_RANDOM') == '1'
        RUN_HIGH = os.environ.get('RUN_HIGH') == '1'
        RUN_LOW = os.environ.get('RUN_LOW') == '1'
        if not (RUN_RANDOM or RUN_HIGH or RUN_LOW):
            RUN_RANDOM = RUN_HIGH = RUN_LOW = True
        
        if BIGPICTURE:
            all_low = default_context.copy()
            all_high = default_context.copy()
            all_random = default_context.copy()
            for feature_name, default_value in default_context.items():
                if not isinstance(default_value, (int, float)) or 'timestep' in feature_name.lower():
                    continue
                all_low[feature_name] = default_value * 0.5
                all_high[feature_name] = default_value * 1.5
                # Random multiplier between 0.5 and 1.5
                all_random[feature_name] = default_value * np.random.uniform(0.5, 1.5)
            if RUN_RANDOM:
                eval_scenarios.append(("All Params Random (0.5x - 1.5x)", all_random))
            if RUN_LOW:
                eval_scenarios.append(("All Params Low (-50%)", all_low))
            if RUN_HIGH:
                eval_scenarios.append(("All Params High (+50%)", all_high))
        else:
            for feature_name, default_value in default_context.items():
                if not isinstance(default_value, (int, float)) or 'timestep' in feature_name.lower():
                    continue
                ctx_low = default_context.copy()
                ctx_low[feature_name] = default_value * 0.5
                
                ctx_high = default_context.copy()
                ctx_high[feature_name] = default_value * 1.5
                
                ctx_random = default_context.copy()
                random_mult = np.random.uniform(0.5, 1.5)
                ctx_random[feature_name] = default_value * random_mult
                
                if RUN_RANDOM:
                    eval_scenarios.append((f"{feature_name} = {ctx_random[feature_name]:.4f} (Random {random_mult:.2f}x)", ctx_random))
                if RUN_LOW:
                    eval_scenarios.append((f"{feature_name} = {ctx_low[feature_name]:.4f} (Low)", ctx_low))
                if RUN_HIGH:
                    eval_scenarios.append((f"{feature_name} = {ctx_high[feature_name]:.4f} (High)", ctx_high))
                
        for mod_label, ctx_dict in eval_scenarios:
            print(colored(f'Evaluating {mod_label}', 'cyan'))
            contexts = {0: ctx_dict}
            
            try:
                env = make_carl_env(cfg, domain, task, contexts=contexts)
            except ValueError as e:
                print(colored(f"  Skipping scenario due to physics constraints: {e}", "red"))
                continue
                
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
            
            print(colored(f'  Result -> R: {np.mean(ep_rewards):.01f} | S: {np.mean(ep_successes):.02f}', 'green'))

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
    evaluate_carl()

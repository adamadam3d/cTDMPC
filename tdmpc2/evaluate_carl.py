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
from carl.envs import CARLDmcWalkerEnv, CARLDmcFishEnv, CARLDmcFingerEnv

# Mapping domains to their corresponding CARL environments
CARL_ENV_MAP = {
    'walker': CARLDmcWalkerEnv,
    'fish': CARLDmcFishEnv,
    'finger': CARLDmcFingerEnv
}

from envs.dmcontrol import DMControlWrapper, suite
from dm_control.suite.wrappers import action_scale
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
    env = action_scale.Wrapper(env, minimum=-1., maximum=1.)
    env = DMControlWrapper(env, domain)
    env = Timeout(env, max_episode_steps=500)
    env = TensorWrapper(env)
    
    try:
        cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
    except:
        cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
    
    cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = env.max_episode_steps
    cfg.seed_steps = max(1000, 5*cfg.episode_length)
    
    return env

@hydra.main(config_name='config', config_path='.')
def evaluate_carl(cfg: dict):
    assert torch.cuda.is_available(), "CUDA required for TD-MPC2."
    assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    
    # Subset of mt30 for walker, fish, and finger
    target_tasks = [
        'walker-stand', 'walker-walk', 'walker-run', 'walker-walk-backwards', 'walker-run-backwards',
        'fish-swim',
        'finger-spin', 'finger-turn-easy', 'finger-turn-hard'
    ]
    
    print(colored('Evaluating CARL modified environments for walker, fish, and finger tasks.', 'yellow', attrs=['bold']))
    
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
                print(colored(f'Task {task_str} not found in cfg.tasks. Skipping.', 'red'))
                continue
        
        print(colored(f'\n--- Task: {task_str} ---', 'magenta', attrs=['bold']))
        
        # Instantiate once to extract the default contexts available for this domain/task
        carl_env_cls = CARL_ENV_MAP[domain]
        temp_env = carl_env_cls(task=task)
        # Use default context from the CARL environment
        default_context = temp_env.get_default_context()
        
        # Iterate and change every single context feature
        for feature_name, default_value in default_context.items():
            if not isinstance(default_value, (int, float)):
                continue
                
            # We vary the default context by evaluating a low (-50%) and high (+50%) value
            modifications = [
                ("Low", default_value * 0.5), 
                ("High", default_value * 1.5)
            ]
            
            for mod_label, mod_val in modifications:
                # Prepare a context dict overriding the specific feature
                contexts = {0: default_context.copy()}
                contexts[0][feature_name] = mod_val
                
                print(colored(f'Evaluating {feature_name} = {mod_val:.4f} ({mod_label})', 'cyan'))
                
                env = make_carl_env(cfg, domain, task, contexts=contexts)
                
                ep_rewards, ep_successes = [], []
                for i in range(cfg.eval_episodes):
                    obs, done, ep_reward, t = env.reset(), False, 0, 0
                    while not done:
                        # Agent act uses task_idx for encoders like task_id
                        action = agent.act(obs, t0=t==0, task=task_idx)
                        prev_obs = obs
                        obs, reward, done, info = env.step(action)
                        # Multi-task context encoders require updating context
                        if getattr(cfg, 'multitask', False):
                            agent.update_context(prev_obs, action, reward, obs)
                        ep_reward += reward
                        t += 1
                    
                    ep_rewards.append(ep_reward)
                    ep_successes.append(info.get('success', 0.0))
                
                print(colored(f'  Result -> R: {np.mean(ep_rewards):.01f} | S: {np.mean(ep_successes):.02f}', 'green'))

if __name__ == '__main__':
    evaluate_carl()

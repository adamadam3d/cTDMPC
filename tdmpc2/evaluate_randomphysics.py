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

# Importing envs.dmcontrol registers the repo's custom dm_control tasks
# (suite.ALL_TASKS is extended at import time) and exposes the base wrappers.
from dm_control import suite
from dm_control.suite.wrappers import action_scale
from envs.dmcontrol import DMControlWrapper, Pixels
from envs.wrappers.timeout import Timeout
from envs.wrappers.tensor import TensorWrapper
from envs import make_env as make_original_env

torch.backends.cudnn.benchmark = True


def randomize_physics(physics, rng, low=0.5, high=1.5, verbose=True):
    """
    Randomly perturb a broad set of MuJoCo physics parameters in-place, CARL-free.

    Each listed model field is multiplied element-wise by an independent factor in
    [low, high], drawn from the provided seeded RNG so runs are reproducible.
    Structurally sensitive fields (timestep, geom sizes, body positions/inertia) use a
    tighter range to avoid destabilizing the simulation. Fields absent for a given
    domain (e.g. tendons in walker) are silently skipped.
    """
    model = physics.model
    # Tighter range for fields that strongly affect stability / kinematics.
    tight_low = 1.0 - (1.0 - low) * 0.4   # e.g. 0.5 -> 0.8
    tight_high = 1.0 + (high - 1.0) * 0.4  # e.g. 1.5 -> 1.2

    def scale(field, lo=low, hi=high):
        """Multiply model.<field> element-wise by random factors; skip if missing/empty."""
        try:
            arr = getattr(model, field)
        except (AttributeError, KeyError):
            return
        if arr is None or np.size(arr) == 0:
            return
        arr[:] *= rng.uniform(lo, hi, size=arr.shape)
        if verbose:
            print(colored(f'    randomized {field:<18} shape={tuple(arr.shape)}', 'grey'))

    with physics.reset_context():  # recompute derived quantities after editing the model
        # --- Global simulation options ---
        model.opt.gravity[:] *= rng.uniform(low, high)              # gravity strength
        model.opt.wind[:] += rng.uniform(-1.0, 1.0, size=3)         # additive wind (base is 0)
        model.opt.density *= rng.uniform(low, high)                 # medium density -> drag
        model.opt.viscosity *= rng.uniform(low, high)              # medium viscosity (swimmers)
        model.opt.timestep *= rng.uniform(tight_low, tight_high)   # integration step (sensitive)

        # --- Per-body inertial properties ---
        scale('body_mass')                    # mass (worldbody index 0 stays 0)
        scale('body_inertia')                 # diagonal inertia tensor
        scale('body_ipos', tight_low, tight_high)  # center-of-mass offset (sensitive)

        # --- Per-joint / per-DOF ---
        scale('dof_damping')                  # joint damping
        scale('dof_frictionloss')             # dry (Coulomb) joint friction
        scale('dof_armature')                 # reflected motor inertia
        scale('jnt_stiffness')                # joint spring stiffness

        # --- Per-geom contact / shape ---
        scale('geom_friction')                # sliding / torsional / rolling friction
        scale('geom_size', tight_low, tight_high)   # geom dimensions (sensitive)
        scale('geom_solref', tight_low, tight_high)  # contact softness / time constant
        scale('geom_gap', tight_low, tight_high)     # contact activation distance

        # --- Per-actuator ---
        scale('actuator_gainprm')             # actuator gain -> force/torque strength
        scale('actuator_forcerange')          # force saturation limits

        # --- Per-tendon (absent in most domains) ---
        scale('tendon_stiffness')
        scale('tendon_damping')


def make_randomized_env(cfg, rng):
    """
    Build a dm_control environment (mirroring envs/dmcontrol.make_env) whose physics
    parameters are randomized once, up front, with the given seeded RNG.
    """
    domain, task = cfg.task.replace('-', '_').split('_', 1)
    domain = dict(cup='ball_in_cup', pointmass='point_mass').get(domain, domain)
    if (domain, task) not in suite.ALL_TASKS:
        raise ValueError('Unknown task:', task)
    assert cfg.obs in {'state', 'rgb'}, 'This task only supports state and rgb observations.'

    env = suite.load(domain, task, task_kwargs={'random': cfg.seed}, visualize_reward=False)
    env = action_scale.Wrapper(env, minimum=-1., maximum=1.)

    # Randomize the underlying MuJoCo model before wrapping.
    randomize_physics(env.physics, rng)

    env = DMControlWrapper(env, domain)
    if cfg.obs == 'rgb':
        env = Pixels(env, cfg)
    env = Timeout(env, max_episode_steps=500)
    env = TensorWrapper(env)
    return env


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
    """
    Script for evaluating a single-task / multi-task TD-MPC2 checkpoint on dm_control
    environments whose physics parameters are randomized with a fixed seed -- WITHOUT
    CARL. Physics is perturbed by editing the MuJoCo model directly (masses, damping,
    friction, gravity).

    Mirrors evaluate.py. The perturbation is drawn from a seeded RNG so runs are
    reproducible.

    Most relevant args:
        `task`: task name (or mt30/mt80 for multi-task evaluation)
        `model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
        `checkpoint`: path to model checkpoint to load
        `eval_episodes`: number of episodes to evaluate on per task (default: 10)
        `save_video`: whether to save a video of the evaluation (default: True)
        `seed`: random seed, also seeds the physics randomization (default: 1)

    Example usage:
    ````
        $ python evaluate_randomphysics.py task=walker-run checkpoint=/path/to/walker.pt
        $ python evaluate_randomphysics.py task=mt30 model_size=317 checkpoint=/path/to/mt30-317M.pt
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

        # Build a randomized-physics env for this task.
        cfg.task = task
        try:
            env = make_randomized_env(cfg, rng)
        except (ValueError, AssertionError) as e:
            print(colored(f'  {task:<22}\tSkipped: {e}', 'red'))
            continue

        ep_rewards, ep_successes = [], []
        for i in range(cfg.eval_episodes):
            obs, done, ep_reward, t = env.reset(), False, 0, 0
            if cfg.save_video:
                frames = [env.render()]
            while not done:
                action = agent.act(obs, t0=t == 0, task=task_idx)
                prev_obs = obs
                obs, reward, done, info = env.step(action)
                if cfg.multitask:
                    agent.update_context(prev_obs, action, reward, obs)
                ep_reward += reward
                t += 1
                if cfg.save_video:
                    frames.append(env.render())
            ep_rewards.append(ep_reward)
            ep_successes.append(info['success'])
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

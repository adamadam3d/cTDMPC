import os
from copy import deepcopy
from time import time
from pathlib import Path
from glob import glob

import numpy as np
import torch
from tqdm import tqdm

from common.buffer import Buffer
from trainer.base import Trainer


class OfflineTrainer(Trainer):
	"""Trainer class for multi-task offline TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._start_time = time()
	
	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		results = dict()
		scores = []
		for task_idx in tqdm(range(len(self.cfg.tasks)), desc='Evaluating'):
			task = self.cfg.tasks[task_idx]
			ep_rewards, ep_successes = [], []
			for _ in range(self.cfg.eval_episodes):
				obs, done, ep_reward, t = self.env.reset(task_idx), False, 0, 0
				while not done:
					torch.compiler.cudagraph_mark_step_begin()
					action = self.agent.act(obs, t0=t==0, eval_mode=True, task=task_idx)
					prev_obs = obs
					obs, reward, done, info = self.env.step(action)
					self.agent.update_context(prev_obs, action, reward, obs)
					ep_reward += reward
					t += 1
				ep_rewards.append(ep_reward)
				ep_successes.append(info['success'])
			ep_reward, ep_success = np.nanmean(ep_rewards), np.nanmean(ep_successes)
			results.update({
				f'episode_reward+{task}': ep_reward,
				f'episode_success+{task}': ep_success,})
			# Per-task normalized score (success for Meta-World, reward/10 otherwise),
			# matching the convention used by evaluate.py.
			scores.append(ep_success*100 if task.startswith('mw-') else ep_reward/10)
		# Aggregate and plateau-monitoring statistics over the normalized scores.
		scores = np.array(scores, dtype=np.float64)
		results['score'] = float(np.nanmean(scores))
		results['score_min'] = float(np.nanmin(scores))
		k = min(5, len(scores))
		results[f'score_bottom{k}'] = float(np.sort(scores)[:k].mean())
		return results
	
	def _load_dataset(self, episodes_per_task=None):
		"""
		Load dataset for offline training.

		If `episodes_per_task` is set, only the first `episodes_per_task`
		episodes of each task are kept. That is plenty for the gradient-conflict
		diagnostics in evaluate_checkpoints.py (which only sample a few
		batch_size-sized batches) and avoids loading the full 345M/550M-transition
		dataset; the deterministic selection also keeps the diagnostic data
		identical across checkpoints and invocations. Data files are
		memory-mapped in this mode so only the selected episodes are read into
		memory.
		"""
		fp = Path(os.path.join(self.cfg.data_dir, '*.pt'))
		fps = sorted(glob(str(fp)))
		assert len(fps) > 0, f'No data found at {fp}'
		print(f'Found {len(fps)} files in {fp}')
		if len(fps) < (20 if self.cfg.task == 'mt80' else 4):
			print(f'WARNING: expected 20 files for mt80 task set, 4 files for mt30 task set, found {len(fps)} files.')

		# Create buffer for sampling
		_cfg = deepcopy(self.cfg)
		_cfg.episode_length = 101 if self.cfg.task == 'mt80' else 501
		if episodes_per_task:
			quota = torch.full((len(self.cfg.tasks),), episodes_per_task, dtype=torch.int64)
			_cfg.buffer_size = episodes_per_task * len(self.cfg.tasks) * _cfg.episode_length
		else:
			_cfg.buffer_size = 550_450_000 if self.cfg.task == 'mt80' else 345_690_000
		_cfg.steps = _cfg.buffer_size
		self.buffer = Buffer(_cfg)
		for fp in tqdm(fps, desc='Loading data'):
			if episodes_per_task:
				if not quota.any():
					break
				try:
					td = torch.load(fp, weights_only=False, mmap=True)
				except Exception:
					td = torch.load(fp, weights_only=False)
			else:
				td = torch.load(fp, weights_only=False)
			assert td.shape[1] == _cfg.episode_length, \
				f'Expected episode length {td.shape[1]} to match config episode length {_cfg.episode_length}, ' \
				f'please double-check your config.'
			if episodes_per_task:
				ep_task = td['task'][:, 0].long()
				keep = []
				for ti in ep_task.unique().tolist():
					if quota[ti] > 0:
						idx = (ep_task == ti).nonzero(as_tuple=True)[0][:int(quota[ti])]
						keep.append(idx)
						quota[ti] -= idx.numel()
				if not keep:
					continue
				td = td[torch.cat(keep).sort().values]
			self.buffer.load(td)
		if episodes_per_task:
			short = int((quota > 0).sum())
			if short:
				print(f'WARNING: {short} task(s) have fewer than {episodes_per_task} episodes in the dataset.')
			print(f'Loaded {self.buffer.num_eps} episodes ({episodes_per_task} per task) for gradient-conflict metrics.')
		else:
			expected_episodes = _cfg.buffer_size // _cfg.episode_length
			if self.buffer.num_eps != expected_episodes:
				print(f'WARNING: buffer has {self.buffer.num_eps} episodes, expected {expected_episodes} episodes for {self.cfg.task} task set.')

	def train(self):
		"""Train a TD-MPC2 agent."""
		assert self.cfg.multitask and self.cfg.task in {'mt30', 'mt80'}, \
			'Offline training only supports multitask training with mt30 or mt80 task sets.'
		self._load_dataset()
		
		start_step = 0
		if getattr(self.cfg, 'checkpoint', None) and self.cfg.checkpoint != '???':
			print(f"Resuming from checkpoint: {self.cfg.checkpoint}")
			start_step = self.agent.load(self.cfg.checkpoint, resume=True)
			print(f"Resuming from iteration {start_step}")

		print(f'Training agent for {self.cfg.steps} iterations...')
		metrics = {}
		for i in range(start_step, self.cfg.steps):

			# Update agent
			train_metrics = self.agent.update(self.buffer)

			# Evaluate agent periodically
			if i % self.cfg.eval_freq == 0 or i % 10_000 == 0:
				metrics = {
					'iteration': i,
					'elapsed_time': time() - self._start_time,
				}
				metrics.update(train_metrics)
				if i % self.cfg.eval_freq == 0:
					metrics.update(self.eval())
					if self.cfg.get('log_grad_conflict', True):
						metrics.update(self.agent.grad_conflict(self.buffer))
					self.logger.pprint_multitask(metrics, self.cfg)
				self.logger.log(metrics, 'pretrain')

			# Save checkpoints on their own cadence, decoupled from eval_freq so
			# eval can be made sparse without losing intermediate checkpoints.
			if i > 0 and i % self.cfg.get('save_freq', self.cfg.eval_freq) == 0:
				self.logger.save_agent(self.agent, identifier=f'{i}', step=i)

		self.logger.finish(self.agent, step=self.cfg.steps)

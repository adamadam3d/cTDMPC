import random

import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.layers import api_model_conversion
from tensordict import TensorDict


class TDMPC2(torch.nn.Module):
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		self.model = WorldModel(cfg).to(self.device)
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._termination.parameters() if self.cfg.episodic else []},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._ctx_enc.parameters() if self.cfg.multitask else []
			 }
		], lr=self.cfg.lr, capturable=True)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		print('Episode length:', cfg.episode_length)
		print('Discount factor:', self.discount)
		self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))
		if cfg.multitask:
			# Online context state for inference during rollouts, per encoder.
			ctx_dim = 2*cfg.obs_shape['state'][0] + cfg.action_dim + 1
			if cfg.context_encoder == 'varibad':
				# Recurrent belief: GRU hidden state carried across the episode.
				self._ctx_h = torch.zeros(1, 1, cfg.enc_dim, device=self.device)
				self._belief = torch.zeros(1, cfg.task_dim, device=self.device)
			else:
				# pearl / supervised: a rolling buffer of recent transitions.
				self._ctx_buf = torch.zeros(cfg.context_window, ctx_dim, device=self.device)
				self._ctx_count = 0
				if cfg.context_encoder == 'pearl':
					self._z_ctx_mu = torch.zeros(1, cfg.task_dim, device=self.device)
					self._z_ctx_logvar = torch.zeros(1, cfg.task_dim, device=self.device)
				else:  # supervised
					self._z_ctx = torch.zeros(1, cfg.task_dim, device=self.device)
		if cfg.compile:
			print('Compiling update function with torch.compile...')
			self._update = torch.compile(self._update, mode="reduce-overhead")

	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		if self.cfg.compile:
			plan = torch.compile(self._plan, mode="reduce-overhead")
		else:
			plan = self._plan
		self._plan_val = plan
		return self._plan_val

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def _named_optim_state(self, optim):
		"""Convert optimizer state from positional to name-keyed, for robust checkpointing."""
		param_to_name = {p: n for n, p in self.model.named_parameters()}
		state = {}
		for group in optim.param_groups:
			for p in group['params']:
				if p in optim.state:
					state[param_to_name[p]] = optim.state[p]
		return state

	def _load_named_optim_state(self, optim, named_state):
		"""Restore name-keyed optimizer state, validating shapes."""
		name_to_param = dict(self.model.named_parameters())
		optim_params = {p for group in optim.param_groups for p in group['params']}
		for name, s in named_state.items():
			p = name_to_param.get(name)
			if p is None:
				raise KeyError(f'Optimizer state for unknown parameter: {name}')
			if p not in optim_params:
				print(f'WARNING: skipping optimizer state for {name} (not in this optimizer; config changed?)')
				continue
			for k, v in s.items():
				if torch.is_tensor(v) and v.ndim > 0 and v.shape != p.shape:
					raise ValueError(
						f'Optimizer state shape mismatch for {name} ({k}): '
						f'{tuple(v.shape)} vs {tuple(p.shape)}')
			optim.state[p] = {k: v.to(p.device) if torch.is_tensor(v) else v
							  for k, v in s.items()}

	def save(self, fp, step=None):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
			step (int): Optional step/iteration to save.
		"""
		payload = {
			"model": self.model.state_dict(),
			"scale": self.scale.state_dict(),
			"cfg_check": {
				"model_size": self.cfg.get("model_size", None),
				"episodic": self.cfg.get("episodic", None),
				"multitask": self.cfg.get("multitask", None),
			},
		}
		if hasattr(self, 'optim'):
			payload["optim_named"] = self._named_optim_state(self.optim)
		if hasattr(self, 'pi_optim'):
			payload["pi_optim_named"] = self._named_optim_state(self.pi_optim)
		if step is not None:
			payload["step"] = step
		torch.save(payload, fp)

	def load(self, fp, resume=False):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str or dict): Filepath or state dict to load.
			resume (bool): Also restore optimizer/scale state to resume training.
		"""
		if isinstance(fp, dict):
			state_dict = fp
		else:
			state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)

		model_state_dict = state_dict["model"] if "model" in state_dict else state_dict
		model_state_dict = api_model_conversion(self.model.state_dict(), model_state_dict)
		self.model.load_state_dict(model_state_dict)

		if resume:
			cfg_check = state_dict.get("cfg_check", None)
			if cfg_check is not None:
				for key, saved in cfg_check.items():
					current = self.cfg.get(key, None)
					if saved is not None and saved != current:
						raise ValueError(
							f'Checkpoint was saved with {key}={saved}, '
							f'but current run has {key}={current}.')
			if "optim_named" in state_dict:
				if hasattr(self, 'optim'):
					self._load_named_optim_state(self.optim, state_dict["optim_named"])
				if hasattr(self, 'pi_optim') and "pi_optim_named" in state_dict:
					self._load_named_optim_state(self.pi_optim, state_dict["pi_optim_named"])
			else:
				print('Checkpoint has no name-keyed optimizer state — resuming weights only.')
			if "scale" in state_dict:
				self.scale.load_state_dict(state_dict["scale"])

		return state_dict.get("step", 0)

	def _reset_context(self):
		"""Reset the online context to its episode-start prior."""
		if self.cfg.context_encoder == 'varibad':
			self._ctx_h.zero_()
			self._belief.zero_()
		else:
			self._ctx_count = 0
			if self.cfg.context_encoder == 'pearl':
				self._z_ctx_mu.zero_()
				self._z_ctx_logvar.zero_()
			else:  # supervised
				self._z_ctx.zero_()

	@torch.no_grad()
	def update_context(self, obs, action, reward, next_obs):
		"""
		Feed a transition into the online context and recompute the task latent.
		Call after each env step. For the recurrent VariBAD encoder this is a
		one-step belief update; for pearl / supervised the transition is appended
		to a rolling buffer and the latent is re-inferred from the window.

		Args:
			obs (torch.Tensor): Observation before the step (padded dims).
			action (torch.Tensor): Action taken (padded dims).
			reward (float): Reward received.
			next_obs (torch.Tensor): Observation after the step (padded dims).
		"""
		tup = torch.cat([
			obs.to(self.device, non_blocking=True).view(-1),
			action.to(self.device, non_blocking=True).view(-1),
			torch.as_tensor(reward, dtype=torch.float32, device=self.device).view(1),
			next_obs.to(self.device, non_blocking=True).view(-1),
		])
		if self.cfg.context_encoder == 'varibad':
			belief, h = self.model.belief_update(tup.unsqueeze(0), self._ctx_h)
			self._ctx_h.copy_(h)
			self._belief.copy_(belief)
			return
		self._ctx_buf[self._ctx_count % self.cfg.context_window] = tup
		self._ctx_count += 1
		n = min(self._ctx_count, self.cfg.context_window)
		mask = (torch.arange(self.cfg.context_window, device=self.device) < n).float()
		if self.cfg.context_encoder == 'pearl':
			mu, logvar = self.model.infer_ctx(self._ctx_buf, mask)
			self._z_ctx_mu.copy_(mu.unsqueeze(0))
			self._z_ctx_logvar.copy_(logvar.unsqueeze(0))
		else:  # supervised
			z = self.model.infer_ctx(self._ctx_buf, mask)
			self._z_ctx.copy_(z.unsqueeze(0))

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Select an action by planning in the latent space of the world model.
		In multi-task mode the task latent is inferred from the online context
		(prior at episode start); `task` is only used for env-side plumbing
		(action masks and discounts), never as a network input.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		if self.cfg.multitask:
			if t0:
				self._reset_context()
			if self.cfg.context_encoder == 'varibad':
				z_ctx = self._belief
			elif self.cfg.context_encoder == 'supervised':
				z_ctx = self._z_ctx
			elif eval_mode:
				z_ctx = self._z_ctx_mu
			else:
				z_ctx = self._z_ctx_mu + torch.randn_like(self._z_ctx_mu) * torch.exp(0.5*self._z_ctx_logvar)
		else:
			z_ctx = None
		if self.cfg.mpc:
			return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task, z_ctx=z_ctx).cpu()
		z = self.model.encode(obs, z_ctx)
		action, info = self.model.pi(z, z_ctx, task)
		if eval_mode:
			action = info["mean"]
		return action[0].cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, task, z_ctx):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], z_ctx), self.cfg)
			z = self.model.next(z, actions[t], z_ctx)
			G = G + discount * (1-termination) * reward
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
			if self.cfg.episodic:
				termination = torch.clip(termination + (self.model.termination(z, z_ctx) > 0.5).float(), max=1.)
		action, _ = self.model.pi(z, z_ctx, task)
		return G + discount * (1-termination) * self.model.Q(z, action, z_ctx, return_type='avg')

	@torch.no_grad()
	def _plan(self, obs, t0=False, eval_mode=False, task=None, z_ctx=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for action masks and discounts).
			z_ctx (torch.Tensor): Task latent inferred from the online context.

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Sample policy trajectories
		z = self.model.encode(obs, z_ctx)
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon-1):
				pi_actions[t], _ = self.model.pi(_z, z_ctx, task)
				_z = self.model.next(_z, pi_actions[t], z_ctx)
			pi_actions[-1], _ = self.model.pi(_z, z_ctx, task)

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions
			r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, self.cfg.num_pi_trajs:] = actions_sample
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions
			value = self._estimate_value(z, actions, task, z_ctx).nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0).values
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
			std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action
		rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		a, std = actions[0], std[0]
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		self._prev_mean.copy_(mean)
		return a.clamp(-1, 1)

	def update_pi(self, zs, z_ctx, task):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			z_ctx (torch.Tensor): Task latent (detached, multi-task only).
			task (torch.Tensor): Task index (only used for action masks).

		Returns:
			float: Loss of the policy update.
		"""
		action, info = self.model.pi(zs, z_ctx, task)
		qs = self.model.Q(zs, action, z_ctx, return_type='avg', detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		info = TensorDict({
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info["entropy"],
			"pi_scaled_entropy": info["scaled_entropy"],
			"pi_scale": self.scale.value,
		})
		return info

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated, task, z_ctx):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			terminated (torch.Tensor): Termination signal at the current time step.
			task (torch.Tensor): Task index (only used for action masks and discounts).
			z_ctx (torch.Tensor): Task latent (multi-task only).

		Returns:
			torch.Tensor: TD-target.
		"""
		action, _ = self.model.pi(next_z, z_ctx, task)
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * (1-terminated) * self.model.Q(next_z, action, z_ctx, return_type='min', target=True)

	def _update(self, obs, action, reward, terminated, task=None, ctx=None, ctx2=None):
		# Switch to train mode before any forward pass that participates in the
		# backward graph. The varibad belief rollout uses a cuDNN RNN, whose
		# backward can only be called if its forward ran in training mode.
		self.model.train()

		# Infer the task latent from context, per encoder.
		ctx_mu = ctx_logvar = beliefs = None
		if not self.cfg.multitask:
			z_ctx = None
		elif self.cfg.context_encoder == 'pearl':
			ctx_mu, ctx_logvar = self.model.infer_ctx(ctx)
			z_ctx = ctx_mu + torch.randn_like(ctx_mu) * torch.exp(0.5*ctx_logvar)
		elif self.cfg.context_encoder == 'varibad':
			# Condition on the belief after a random number of context steps
			# (including 0 = prior), exposing the model to the full spectrum of
			# belief widths encountered online (Bayes-adaptive training).
			beliefs = self.model.belief_rollout(ctx)
			t_idx = torch.randint(0, beliefs.shape[1], (beliefs.shape[0],), device=beliefs.device)
			z_ctx = beliefs.gather(1, t_idx.view(-1, 1, 1).expand(-1, 1, beliefs.shape[-1])).squeeze(1)
		else:  # supervised
			z_ctx = self.model.infer_ctx(ctx)

		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], z_ctx)
			td_targets = self._td_target(next_z, reward, terminated, task, z_ctx)

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], z_ctx)
		zs[0] = z
		consistency_loss = 0
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			z = self.model.next(z, _action, z_ctx)
			consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, z_ctx, return_type='all')
		reward_preds = self.model.reward(_zs, action, z_ctx)
		if self.cfg.episodic:
			termination_pred = self.model.termination(zs[1:], z_ctx, unnormalized=True)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

		consistency_loss = consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
		if self.cfg.episodic:
			termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
		else:
			termination_loss = 0.
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		# Context regulariser, per encoder: KL for the probabilistic encoders
		# (pearl: KL to the N(0,I) prior; varibad: sequential KL belief_t || belief_{t-1})
		# or a supervised loss (task classification / supervised InfoNCE).
		kl_loss, context_loss, context_acc = 0., 0., 0.
		if self.cfg.multitask and self.cfg.context_encoder == 'pearl':
			kl_loss = math.gaussian_kl(ctx_mu, ctx_logvar).mean()
			ctx_reg = self.cfg.kl_coef * kl_loss
		elif self.cfg.multitask and self.cfg.context_encoder == 'varibad':
			mus, logvars = beliefs.chunk(2, dim=-1)
			kl_loss = math.gaussian_kl_pair(
				mus[:, 1:], logvars[:, 1:], mus[:, :-1], logvars[:, :-1]).mean()
			ctx_reg = self.cfg.kl_coef * kl_loss
		elif self.cfg.multitask:  # supervised
			if self.cfg.context_loss == 'nce':
				context_loss = math.info_nce(z_ctx, self.model.infer_ctx(ctx2), task, self.cfg.nce_temp)
			else:
				task_logits = self.model.classify_ctx(z_ctx)
				context_loss = F.cross_entropy(task_logits, task)
				context_acc = (task_logits.argmax(-1) == task).float().mean()
			ctx_reg = self.cfg.context_coef * context_loss
		else:
			ctx_reg = 0.
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.termination_coef * termination_loss +
			self.cfg.value_coef * value_loss +
			ctx_reg
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		# Update policy
		pi_info = self.update_pi(zs.detach(), z_ctx.detach() if z_ctx is not None else None, task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		info = TensorDict({
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"termination_loss": termination_loss,
			"kl_loss": kl_loss,
			"context_loss": context_loss,
			"context_acc": context_acc,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
		})
		if self.cfg.episodic:
			info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))
		if self.cfg.multitask and self.cfg.context_encoder in ('pearl', 'varibad'):
			# Posterior diagnostics: average posterior std (-> 0 as the context
			# becomes identifiable, -> 1 as it falls back to the prior) and the
			# typical magnitude of the inferred context mean. For varibad z_ctx is
			# the belief [mu, logvar]; for pearl use the inferred (mu, logvar).
			if self.cfg.context_encoder == 'varibad':
				ctx_mu, ctx_logvar = z_ctx.chunk(2, dim=-1)
			info.update(TensorDict({
				"ctx_post_std": torch.exp(0.5*ctx_logvar).mean(),
				"ctx_mu_norm": ctx_mu.norm(dim=-1).mean(),
			}))
		elif self.cfg.multitask:  # supervised
			info.update(TensorDict({"ctx_emb_norm": z_ctx.norm(dim=-1).mean()}))
		info.update(pi_info)
		return info.detach().mean()

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, terminated, task, ctx = buffer.sample()
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
			kwargs["ctx"] = ctx
			if self.cfg.context_encoder == 'supervised' and self.cfg.context_loss == 'nce':
				# Independent second window per task for contrastive positives.
				kwargs["ctx2"] = buffer.sample_context(task)
		torch.compiler.cudagraph_mark_step_begin()
		return self._update(obs, action, reward, terminated, **kwargs)

	def _model_loss(self, obs, action, reward, terminated, task, ctx):
		"""
		Scalar world-model loss for a (sub-)batch, used only as a diagnostic by
		`grad_conflict`. Mirrors the world-model terms of `_update` (consistency,
		reward, value) plus the encoder's per-task context regulariser, runs
		eagerly (never compiled), and performs no optimiser or policy update.
		The termination term is omitted (inactive for the non-episodic multitask
		suites used here); for the supervised encoder the InfoNCE loss is omitted
		because it couples tasks within a batch (only the CE term is per-task).
		"""
		# Context inference, per encoder.
		ctx_mu = ctx_logvar = beliefs = None
		if self.cfg.context_encoder == 'pearl':
			ctx_mu, ctx_logvar = self.model.infer_ctx(ctx)
			z_ctx = ctx_mu + torch.randn_like(ctx_mu) * torch.exp(0.5*ctx_logvar)
		elif self.cfg.context_encoder == 'varibad':
			beliefs = self.model.belief_rollout(ctx)
			t_idx = torch.randint(0, beliefs.shape[1], (beliefs.shape[0],), device=beliefs.device)
			z_ctx = beliefs.gather(1, t_idx.view(-1, 1, 1).expand(-1, 1, beliefs.shape[-1])).squeeze(1)
		else:  # supervised
			z_ctx = self.model.infer_ctx(ctx)
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], z_ctx)
			td_targets = self._td_target(next_z, reward, terminated, task, z_ctx)
		zs = [self.model.encode(obs[0], z_ctx)]
		consistency_loss = 0
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			zs.append(self.model.next(zs[-1], _action, z_ctx))
			consistency_loss = consistency_loss + F.mse_loss(zs[-1], _next_z) * self.cfg.rho**t
		zs = torch.stack(zs, 0)
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, z_ctx, return_type='all')
		reward_preds = self.model.reward(_zs, action, z_ctx)
		reward_loss, value_loss = 0, 0
		for t, (rew_pred, rew, td_target, qs_t) in enumerate(zip(
				reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred, rew, self.cfg).mean() * self.cfg.rho**t
			for qs_t_q in qs_t.unbind(0):
				value_loss = value_loss + math.soft_ce(qs_t_q, td_target, self.cfg).mean() * self.cfg.rho**t
		consistency_loss = consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.value_coef * value_loss
		)
		# Per-task context regulariser (NCE omitted: it couples tasks).
		if self.cfg.context_encoder == 'pearl':
			loss = loss + self.cfg.kl_coef * math.gaussian_kl(ctx_mu, ctx_logvar).mean()
		elif self.cfg.context_encoder == 'varibad':
			mus, logvars = beliefs.chunk(2, dim=-1)
			loss = loss + self.cfg.kl_coef * math.gaussian_kl_pair(
				mus[:, 1:], logvars[:, 1:], mus[:, :-1], logvars[:, :-1]).mean()
		elif self.cfg.context_loss != 'nce':  # supervised, ce
			loss = loss + self.cfg.context_coef * F.cross_entropy(self.model.classify_ctx(z_ctx), task)
		return loss

	@staticmethod
	def _pair_cos(G):
		"""Pairwise cosine similarities (upper triangle) between rows of `G`."""
		Gn = F.normalize(G, dim=1)
		cos = Gn @ Gn.t()
		n = cos.shape[0]
		iu = torch.triu_indices(n, n, offset=1, device=cos.device)
		return cos[iu[0], iu[1]]

	def grad_conflict(self, buffer, max_tasks=None):
		"""
		Measure destructive gradient interference between tasks (multitask only).

		Samples a batch and computes the world-model gradient separately for each
		of up to `max_tasks` tasks present in the batch, over the *full* set of
		parameters updated by the world-model loss (encoder, dynamics, reward,
		value/Q, and context encoder -- i.e. every shared component except the
		policy prior, which is trained by a separate objective, and the
		no-grad target/detach Q copies). It reports the fraction of task pairs
		whose gradients have negative cosine similarity and the mean pairwise
		cosine, both overall and broken down per module (so one can see *where*
		tasks interfere). A high conflict fraction is direct evidence of negative
		transfer. Cheap enough to call at every evaluation; returns {} when not
		applicable. Does not modify optimiser state.
		"""
		if not self.cfg.multitask:
			return {}
		max_tasks = max_tasks or self.cfg.get('grad_conflict_tasks', 6)
		obs, action, reward, terminated, task, ctx = buffer.sample()
		# Per-module parameter groups updated by the world-model loss, in a fixed
		# order, with the contiguous index range each occupies in the flat vector.
		modules = [
			('encoder', self.model._encoder),
			('dynamics', self.model._dynamics),
			('reward', self.model._reward),
			('Q', self.model._Qs),
			('ctx', self.model._ctx_enc),
		]
		params, ranges, off = [], {}, 0
		for name, mod in modules:
			ps = [p for p in mod.parameters() if p.requires_grad]
			n = sum(p.numel() for p in ps)
			ranges[name] = (off, off + n)
			off += n
			params += ps
		# Group batch indices by task, keeping only tasks with >= 2 samples.
		groups = []
		for ti in task.unique().tolist():
			idx = (task == ti).nonzero(as_tuple=True)[0]
			if idx.numel() >= 2:
				groups.append(idx)
		if len(groups) < 2:
			return {}
		random.shuffle(groups)
		groups = groups[:max_tasks]
		# The recurrent (VariBAD) context encoder uses a cuDNN GRU whose backward
		# can only be called if its forward ran in training mode. grad_conflict is
		# invoked during eval, so switch to train mode for the gradient computation
		# (this also matches how _update computes the world-model gradient) and
		# restore the previous mode afterwards.
		was_training = self.model.training
		self.model.train()
		try:
			grads = []
			for idx in groups:
				loss = self._model_loss(
					obs[:, idx], action[:, idx], reward[:, idx], terminated[:, idx],
					task[idx], ctx[idx])
				g = torch.autograd.grad(loss, params, allow_unused=True)
				grads.append(torch.cat([
					(gi if gi is not None else torch.zeros_like(p)).reshape(-1)
					for gi, p in zip(g, params)]))
		finally:
			self.model.train(was_training)
		G = torch.stack(grads, 0)
		# Overall conflict over the full world-model gradient.
		pair_cos = self._pair_cos(G)
		out = {
			'grad_conflict_frac': (pair_cos < 0).float().mean().item(),
			'grad_cos_mean': pair_cos.mean().item(),
			'grad_conflict_ntasks': float(G.shape[0]),
		}
		# Per-module breakdown: where do the tasks fight?
		for name, (lo, hi) in ranges.items():
			pc = self._pair_cos(G[:, lo:hi])
			out[f'grad_conflict_frac/{name}'] = (pc < 0).float().mean().item()
			out[f'grad_cos_mean/{name}'] = pc.mean().item()
		return out

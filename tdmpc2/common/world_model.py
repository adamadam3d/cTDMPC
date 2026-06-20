from copy import deepcopy

import torch
import torch.nn as nn

from common import layers, math, init
from tensordict import TensorDict
from tensordict.nn import TensorDictParams


class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		if cfg.multitask:
			# VariBAD-style recurrent context encoder: a GRU processes context
			# transitions (s, a, r, s') in temporal order and parameterizes a
			# posterior q(m | tau_{:t}) over the latent task variable at every
			# step. Networks are conditioned on the belief b_t = [mu_t, logvar_t],
			# so the latent task variable has dimension task_dim // 2.
			assert cfg.task_dim % 2 == 0, 'task_dim must be even (belief is [mu, logvar])'
			ctx_dim = 2*cfg.obs_shape['state'][0] + cfg.action_dim + 1
			self._ctx_enc = nn.ModuleDict({
				'embed': layers.NormedLinear(ctx_dim, cfg.enc_dim),
				'gru': nn.GRU(cfg.enc_dim, cfg.enc_dim, batch_first=True),
				'head': nn.Linear(cfg.enc_dim, cfg.task_dim),
			})
			self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		self._encoder = layers.enc(cfg)
		self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._termination = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 1) if cfg.episodic else None
		self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		self.apply(init.weight_init)
		init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])

		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
		self.init()

	def init(self):
		# Create params
		self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
		self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

		# Create modules
		with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
			self._detach_Qs = deepcopy(self._Qs)
			self._target_Qs = deepcopy(self._Qs)

		# Assign params to modules
		# We do this strange assignment to avoid having duplicated tensors in the state-dict -- working on a better API for this
		delattr(self._detach_Qs, "params")
		self._detach_Qs.__dict__["params"] = self._detach_Qs_params
		delattr(self._target_Qs, "params")
		self._target_Qs.__dict__["params"] = self._target_Qs_params

	def __repr__(self):
		repr = 'TD-MPC2 World Model\n'
		modules = ['Encoder', 'Dynamics', 'Reward', 'Termination', 'Policy prior', 'Q-functions']
		for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._termination, self._pi, self._Qs]):
			if m == self._termination and not self.cfg.episodic:
				continue
			repr += f"{modules[i]}: {m}\n"
		repr += "Learnable parameters: {:,}".format(self.total_params)
		return repr

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self.init()
		return self

	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		super().train(mode)
		self._target_Qs.train(False)
		return self

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

	def _belief(self, hidden):
		"""Project GRU hidden states to belief vectors [mu, logvar]."""
		mu, logvar = self._ctx_enc['head'](hidden).chunk(2, dim=-1)
		return torch.cat([mu, logvar.clamp(-10, 2)], dim=-1)

	def belief_rollout(self, ctx):
		"""
		Encodes a temporally ordered context window with the recurrent encoder
		and returns the per-step beliefs over the latent task variable, with
		the N(0, I) prior belief prepended at index 0 (VariBAD-style).

		Args:
			ctx (torch.Tensor): Context tuples in temporal order, shape (B, N, ctx_dim).

		Returns:
			torch.Tensor: Belief sequence [mu, logvar], shape (B, N+1, task_dim).
		"""
		x = self._ctx_enc['embed'](ctx)
		x, _ = self._ctx_enc['gru'](x)
		beliefs = self._belief(x)
		prior = torch.zeros_like(beliefs[:, :1])
		return torch.cat([prior, beliefs], dim=1)

	def belief_update(self, tup, h):
		"""
		One-step recurrent belief update for online inference.

		Args:
			tup (torch.Tensor): Single context tuple, shape (B, ctx_dim).
			h (torch.Tensor): GRU hidden state, shape (1, B, enc_dim).

		Returns:
			tuple: Updated belief [mu, logvar] of shape (B, task_dim),
				and the new hidden state.
		"""
		x = self._ctx_enc['embed'](tup).unsqueeze(1)
		x, h = self._ctx_enc['gru'](x, h)
		return self._belief(x[:, -1]), h

	def concat_ctx(self, x, z_ctx):
		"""
		Concatenates the inferred task latent `z_ctx` to the input `x`.
		Broadcasts over the time dimension when `x` is a sequence.
		"""
		if x.ndim == 3:
			z_ctx = z_ctx.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif z_ctx.shape[0] == 1:
			z_ctx = z_ctx.repeat(x.shape[0], 1)
		return torch.cat([x, z_ctx], dim=-1)

	def encode(self, obs, z_ctx):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		if self.cfg.multitask:
			obs = self.concat_ctx(obs, z_ctx)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
		return self._encoder[self.cfg.obs](obs)

	def next(self, z, a, z_ctx):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		if self.cfg.multitask:
			z = self.concat_ctx(z, z_ctx)
		z = torch.cat([z, a], dim=-1)
		return self._dynamics(z)

	def reward(self, z, a, z_ctx):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		if self.cfg.multitask:
			z = self.concat_ctx(z, z_ctx)
		z = torch.cat([z, a], dim=-1)
		return self._reward(z)

	def termination(self, z, z_ctx, unnormalized=False):
		"""
		Predicts termination signal.
		"""
		if self.cfg.multitask:
			z = self.concat_ctx(z, z_ctx)
		if unnormalized:
			return self._termination(z)
		return torch.sigmoid(self._termination(z))


	def pi(self, z, z_ctx, task=None):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		`task` is only used to mask out unused action dimensions
		(env-side knowledge); the networks are conditioned on `z_ctx` alone.
		"""
		if self.cfg.multitask:
			z = self.concat_ctx(z, z_ctx)

		# Gaussian policy prior
		mean, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mean)

		if self.cfg.multitask: # Mask out unused action dimensions
			mean = mean * self._action_masks[task]
			log_std = log_std * self._action_masks[task]
			eps = eps * self._action_masks[task]
			action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
		else: # No masking
			action_dims = None

		log_prob = math.gaussian_logprob(eps, log_std)

		# Scale log probability by action dimensions
		size = eps.shape[-1] if action_dims is None else action_dims
		scaled_log_prob = log_prob * size

		# Reparameterization trick
		action = mean + eps * log_std.exp()
		mean, action, log_prob = math.squash(mean, action, log_prob)

		entropy_scale = scaled_log_prob / (log_prob + 1e-8)
		info = TensorDict({
			"mean": mean,
			"log_std": log_std,
			"action_prob": 1.,
			"entropy": -log_prob,
			"scaled_entropy": -log_prob * entropy_scale,
		})
		return action, info

	def Q(self, z, a, z_ctx, return_type='min', target=False, detach=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}

		if self.cfg.multitask:
			z = self.concat_ctx(z, z_ctx)

		z = torch.cat([z, a], dim=-1)
		if target:
			qnet = self._target_Qs
		elif detach:
			qnet = self._detach_Qs
		else:
			qnet = self._Qs
		out = qnet(z)

		if return_type == 'all':
			return out

		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		Q = math.two_hot_inv(out[qidx], self.cfg)
		if return_type == "min":
			return Q.min(0).values
		return Q.sum(0) / 2

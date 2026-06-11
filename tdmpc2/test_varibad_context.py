"""
CPU unit checks for the VariBAD-style recurrent context encoder
(GRU belief inference over the latent task variable, replacing the
task embedding).

Run with: python test_varibad_context.py
"""
import torch
from tensordict.tensordict import TensorDict

from common import math
from common.world_model import WorldModel
from common.buffer import Buffer


class Cfg:
	"""Minimal config standing in for the parsed Hydra config."""
	def __init__(self, **kwargs):
		for k, v in kwargs.items():
			setattr(self, k, v)

	def get(self, key, default=None):
		return getattr(self, key, default)


def make_mt_cfg():
	return Cfg(
		multitask=True,
		tasks=['a', 'b', 'c'],
		obs='state',
		obs_shape={'state': (8,)},
		action_dim=4,
		action_dims=[4, 2, 3],
		task_dim=16,
		latent_dim=32,
		mlp_dim=32,
		enc_dim=32,
		num_enc_layers=2,
		num_channels=32,
		episodic=False,
		num_bins=5,
		vmin=-10,
		vmax=10,
		num_q=2,
		dropout=0.,
		simnorm_dim=8,
		log_std_min=-10,
		log_std_max=2,
	)


def test_gaussian_kl_pair():
	mu = torch.randn(4, 3)
	logvar = torch.randn(4, 3)
	# KL between identical Gaussians is zero
	assert torch.allclose(math.gaussian_kl_pair(mu, logvar, mu, logvar), torch.zeros(4), atol=1e-6)
	# KL against the N(0, I) prior matches the specialized helper
	zeros = torch.zeros_like(mu)
	assert torch.allclose(
		math.gaussian_kl_pair(mu, logvar, zeros, zeros),
		math.gaussian_kl(mu, logvar), atol=1e-5)
	assert (math.gaussian_kl_pair(mu + 3., logvar, mu, logvar) > 0).all()
	print('gaussian_kl_pair: OK')


def test_belief_rollout():
	cfg = make_mt_cfg()
	model = WorldModel(cfg)
	model.eval()
	B, N = 6, 10
	ctx_dim = 2*cfg.obs_shape['state'][0] + cfg.action_dim + 1
	ctx = torch.randn(B, N, ctx_dim)

	beliefs = model.belief_rollout(ctx)
	assert beliefs.shape == (B, N+1, cfg.task_dim)
	# Index 0 is the N(0, I) prior belief [mu=0, logvar=0]
	assert torch.allclose(beliefs[:, 0], torch.zeros(B, cfg.task_dim))
	# Beliefs depend on the order of the context (recurrent, not a set encoder)
	perm = torch.randperm(N)
	beliefs_perm = model.belief_rollout(ctx[:, perm])
	assert not torch.allclose(beliefs[:, -1], beliefs_perm[:, -1], atol=1e-4)
	print('belief_rollout: OK')


def test_online_offline_consistency():
	"""Step-by-step belief_update must match the batched belief_rollout."""
	cfg = make_mt_cfg()
	model = WorldModel(cfg)
	model.eval()
	N = 8
	ctx_dim = 2*cfg.obs_shape['state'][0] + cfg.action_dim + 1
	ctx = torch.randn(1, N, ctx_dim)

	beliefs = model.belief_rollout(ctx)
	h = torch.zeros(1, 1, cfg.enc_dim)
	for t in range(N):
		belief_t, h = model.belief_update(ctx[:, t], h)
		assert torch.allclose(belief_t, beliefs[:, t+1], atol=1e-5), f'mismatch at step {t}'
	print('online/offline belief consistency: OK')


def test_world_model_multitask():
	cfg = make_mt_cfg()
	model = WorldModel(cfg)
	B, N = 6, 10
	ctx_dim = 2*cfg.obs_shape['state'][0] + cfg.action_dim + 1
	beliefs = model.belief_rollout(torch.randn(B, N, ctx_dim))
	z_ctx = beliefs[:, -1]

	# Forward passes conditioned on the belief
	obs = torch.randn(B, cfg.obs_shape['state'][0])
	z = model.encode(obs, z_ctx)
	assert z.shape == (B, cfg.latent_dim)
	a = torch.randn(B, cfg.action_dim)
	assert model.next(z, a, z_ctx).shape == (B, cfg.latent_dim)
	assert model.reward(z, a, z_ctx).shape == (B, cfg.num_bins)
	task = torch.randint(0, len(cfg.tasks), (B,))
	action, _ = model.pi(z, z_ctx, task)
	assert action.shape == (B, cfg.action_dim)
	# Action masking still applies (env-side task knowledge)
	assert (action * (1 - model._action_masks[task]) == 0).all()
	assert model.Q(z, a, z_ctx, return_type='min').shape == (B, 1)

	# Sequence broadcast (T, B, latent) with z_ctx (B, task_dim)
	zs = torch.randn(4, B, cfg.latent_dim)
	acts = torch.randn(4, B, cfg.action_dim)
	assert model.reward(zs, acts, z_ctx).shape == (4, B, cfg.num_bins)

	# Planning-style broadcast: z_ctx (1, d) repeated across samples
	z_many = torch.randn(64, cfg.latent_dim)
	a_many = torch.randn(64, cfg.action_dim)
	assert model.Q(z_many, a_many, z_ctx[:1], return_type='avg').shape == (64, 1)
	print('world_model multitask: OK')


def test_world_model_singletask():
	cfg = make_mt_cfg()
	cfg.multitask = False
	cfg.tasks = ['a']
	cfg.task_dim = 0
	model = WorldModel(cfg)
	B = 6
	obs = torch.randn(B, cfg.obs_shape['state'][0])
	z = model.encode(obs, None)
	a = torch.randn(B, cfg.action_dim)
	assert model.next(z, a, None).shape == (B, cfg.latent_dim)
	action, _ = model.pi(z, None)
	assert action.shape == (B, cfg.action_dim)
	print('world_model single-task: OK')


def test_buffer_context_sampling():
	num_tasks, eps_per_task, ep_len, obs_dim, action_dim = 3, 4, 20, 4, 2
	cfg = Cfg(
		multitask=True,
		tasks=['a', 'b', 'c'],
		buffer_size=10_000,
		steps=10_000,
		batch_size=8,
		horizon=3,
		num_context=16,
	)
	buffer = Buffer(cfg)
	buffer._device = torch.device('cpu')

	# Synthetic episodes: obs encodes task*1000 + global step, action encodes task,
	# reward encodes task*10, so sampled tuples can be traced back exactly.
	tds, step = [], 0
	for task in range(num_tasks):
		for _ in range(eps_per_task):
			obs = torch.zeros(ep_len, obs_dim)
			action = torch.full((ep_len, action_dim), float(task))
			reward = torch.full((ep_len,), task*10.)
			for t in range(ep_len):
				obs[t] = task*1000 + step
				step += 1
			tds.append(TensorDict({
				'obs': obs, 'action': action, 'reward': reward,
				'task': torch.full((ep_len,), task, dtype=torch.int64),
			}, batch_size=(ep_len,)))
	td = torch.stack(tds)

	from torchrl.data.replay_buffers import LazyTensorStorage
	buffer._storage_device = torch.device('cpu')
	buffer._buffer = buffer._reserve_buffer(LazyTensorStorage(buffer._capacity, device='cpu'))
	episode_idx = torch.arange(len(td), dtype=torch.int64)
	td['episode'] = episode_idx.unsqueeze(-1).expand(-1, ep_len)
	buffer._buffer.extend(td.reshape(td.shape[0]*td.shape[1]))
	buffer._num_eps = len(td)

	tasks = torch.tensor([0, 1, 2, 2, 1, 0])
	ctx = buffer.sample_context(tasks, num_context=cfg.num_context)
	assert ctx.shape == (len(tasks), cfg.num_context, 2*obs_dim + action_dim + 1)
	s = ctx[..., 0]
	a = ctx[..., obs_dim]
	r = ctx[..., obs_dim + action_dim]
	s_next = ctx[..., obs_dim + action_dim + 1]
	expected = tasks.unsqueeze(1).float()
	assert torch.allclose(a, expected), 'context action belongs to wrong task'
	assert torch.allclose(r, expected*10), 'context reward belongs to wrong task'
	assert torch.allclose(s.div(1000).floor(), expected), 'context obs belongs to wrong task'
	assert torch.allclose(s_next - s, torch.ones_like(s)), 's and s\' are not adjacent steps'
	# Windows must be temporally ordered and contiguous within an episode
	assert torch.allclose(s[:, 1:] - s[:, :-1], torch.ones_like(s[:, 1:])), \
		'window steps are not consecutive'
	assert torch.allclose(s_next[:, :-1], s[:, 1:]), 's\'(t) != s(t+1) within window'
	print('buffer context sampling (ordered windows): OK')


if __name__ == '__main__':
	torch.manual_seed(0)
	test_gaussian_kl_pair()
	test_belief_rollout()
	test_online_offline_consistency()
	test_world_model_multitask()
	test_world_model_singletask()
	test_buffer_context_sampling()
	print('\nAll VariBAD context checks passed.')

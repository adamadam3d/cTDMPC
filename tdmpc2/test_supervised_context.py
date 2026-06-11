"""
CPU unit checks for the supervised/contrastive context encoder baseline
(deterministic mean-pooled task latent trained with task labels,
replacing the task embedding; no VAE).

Run with: python test_supervised_context.py
"""
import torch
import torch.nn.functional as F
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


def test_info_nce():
	B, d = 8, 16
	labels = torch.arange(B)
	z = F.normalize(torch.randn(B, d), dim=-1)
	# Aligned pairs (positives identical) should give much lower loss than
	# misaligned pairs (positives shuffled).
	aligned = math.info_nce(z, z.clone(), labels, temperature=0.1)
	shuffled = math.info_nce(z, z[torch.randperm(B)], labels, temperature=0.1)
	assert aligned < shuffled, 'aligned positives should have lower loss'

	# Same-label masking: a duplicate-label element whose embedding equals
	# the anchor must not act as a negative (loss stays low).
	labels_dup = labels.clone()
	labels_dup[1] = labels_dup[0]
	z_dup = z.clone()
	z_dup[1] = z_dup[0]
	masked = math.info_nce(z_dup, z_dup.clone(), labels_dup, temperature=0.1)
	assert torch.isfinite(masked) and masked < shuffled, \
		'same-label duplicates must be masked out of the negatives'
	print('info_nce: OK')


def test_infer_ctx():
	cfg = make_mt_cfg()
	model = WorldModel(cfg)
	model.eval()
	B, N = 6, 10
	ctx_dim = 2*cfg.obs_shape['state'][0] + cfg.action_dim + 1
	ctx = torch.randn(B, N, ctx_dim)

	z = model.infer_ctx(ctx)
	assert z.shape == (B, cfg.task_dim)

	# Fully masked context -> zero latent (the "uninformed" state at t0)
	z0 = model.infer_ctx(torch.randn(N, ctx_dim), torch.zeros(N))
	assert torch.allclose(z0, torch.zeros(cfg.task_dim))

	# Masked mean equals the mean over the unmasked subset
	mask = torch.zeros(B, N)
	mask[:, :3] = 1.
	z_m = model.infer_ctx(ctx, mask)
	z_s = model.infer_ctx(ctx[:, :3])
	assert torch.allclose(z_m, z_s, atol=1e-5)

	# Classification head
	logits = model.classify_ctx(z)
	assert logits.shape == (B, len(cfg.tasks))
	print('infer_ctx + classify_ctx: OK')


def test_world_model_multitask():
	cfg = make_mt_cfg()
	model = WorldModel(cfg)
	B, N = 6, 10
	ctx_dim = 2*cfg.obs_shape['state'][0] + cfg.action_dim + 1
	z_ctx = model.infer_ctx(torch.randn(B, N, ctx_dim))

	# Forward passes conditioned on z_ctx
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
	print('buffer context sampling: OK')


if __name__ == '__main__':
	torch.manual_seed(0)
	test_info_nce()
	test_infer_ctx()
	test_world_model_multitask()
	test_world_model_singletask()
	test_buffer_context_sampling()
	print('\nAll supervised-context checks passed.')

import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler


class Buffer():
	"""
	Replay buffer for TD-MPC2 training. Based on torchrl.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self._device = torch.device('cuda:0')
		self._capacity = min(cfg.buffer_size, cfg.steps)
		self._sampler = SliceSampler(
			num_slices=self.cfg.batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
			cache_values=cfg.multitask,
		)
		self._batch_size = cfg.batch_size * (cfg.horizon+1)
		self._num_eps = 0
		self._ctx_rows = None

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity

	@property
	def num_eps(self):
		"""Return the number of episodes in the buffer."""
		return self._num_eps

	def _reserve_buffer(self, storage):
		"""
		Reserve a buffer with the given storage.
		"""
		return ReplayBuffer(
			storage=storage,
			sampler=self._sampler,
			pin_memory=False,
			prefetch=0,
			batch_size=self._batch_size,
		)

	def _init(self, tds):
		"""Initialize the replay buffer. Use the first episode to estimate storage requirements."""
		print(f'Buffer capacity: {self._capacity:,}')
		mem_free, _ = torch.cuda.mem_get_info()
		bytes_per_step = sum([
				(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel()*x.element_size() for x in v.values()])) \
			for v in tds.values()
		]) / len(tds)
		total_bytes = bytes_per_step*self._capacity
		print(f'Storage required: {total_bytes/1e9:.2f} GB')
		# Heuristic: decide whether to use CUDA or CPU memory
		storage_device = 'cuda:0' if 2.5*total_bytes < mem_free else 'cpu'
		print(f'Using {storage_device.upper()} memory for storage.')
		self._storage_device = torch.device(storage_device)
		return self._reserve_buffer(
			LazyTensorStorage(self._capacity, device=self._storage_device)
		)

	def load(self, td):
		"""
		Load a batch of episodes into the buffer. This is useful for loading data from disk,
		and is more efficient than adding episodes one by one.
		"""
		num_new_eps = len(td)
		episode_idx = torch.arange(self._num_eps, self._num_eps+num_new_eps, dtype=torch.int64)
		td['episode'] = episode_idx.unsqueeze(-1).expand(-1, td['reward'].shape[1])
		if self._num_eps == 0:
			self._buffer = self._init(td[0])
		td = td.reshape(td.shape[0]*td.shape[1])
		self._buffer.extend(td)
		self._num_eps += num_new_eps
		return self._num_eps

	def add(self, td):
		"""Add an episode to the buffer."""
		td['episode'] = torch.full_like(td['reward'], self._num_eps, dtype=torch.int64)
		if self._num_eps == 0:
			self._buffer = self._init(td)
		self._buffer.extend(td)
		self._num_eps += 1
		return self._num_eps

	def _prepare_batch(self, td):
		"""
		Prepare a sampled batch for training (post-processing).
		Expects `td` to be a TensorDict with batch size TxB.
		"""
		td = td.select("obs", "action", "reward", "terminated", "task", strict=False).to(self._device, non_blocking=True)
		obs = td.get('obs').contiguous()
		action = td.get('action')[1:].contiguous()
		reward = td.get('reward')[1:].unsqueeze(-1).contiguous()
		terminated = td.get('terminated', None)
		if terminated is not None:
			terminated = td.get('terminated')[1:].unsqueeze(-1).contiguous()
		else:
			terminated = torch.zeros_like(reward)
		task = td.get('task', None)
		if task is not None:
			task = task[0].contiguous()
		return obs, action, reward, terminated, task

	def _build_ctx_index(self):
		"""
		Build a flat per-task index over storage rows for context sampling.
		Valid rows are non-episode-first steps, so that row `i` holds
		(a, r, s'=obs_i) and row `i-1` holds s=obs_{i-1} of the same episode.
		"""
		storage = self._buffer._storage
		n = len(storage)
		td = storage[:n]
		episode = td.get('episode').view(n).long()
		task = td.get('task').view(n).long()
		valid = torch.zeros(n, dtype=torch.bool, device=episode.device)
		valid[1:] = episode[1:] == episode[:-1]
		rows = valid.nonzero(as_tuple=True)[0]
		order = torch.argsort(task[rows], stable=True)
		self._ctx_rows = rows[order].to(torch.int32)
		counts = torch.bincount(task[rows], minlength=len(self.cfg.tasks))
		self._ctx_offsets = torch.cat([torch.zeros(1, dtype=torch.int64, device=counts.device), counts.cumsum(0)])

	def sample_context(self, tasks, num_context=None):
		"""
		Sample `num_context` (s, a, r, s') context tuples per task in `tasks`,
		drawn uniformly from the data of that task.

		Args:
			tasks (torch.Tensor): Task IDs of shape (B,).
			num_context (int): Context transitions per batch element.

		Returns:
			torch.Tensor: Context batch of shape (B, num_context, ctx_dim).
		"""
		if self._ctx_rows is None:
			self._build_ctx_index()
		num_context = num_context or self.cfg.num_context
		device = self._ctx_offsets.device
		tasks = tasks.to(device).long()
		starts = self._ctx_offsets[tasks]
		counts = self._ctx_offsets[tasks+1] - starts
		rand = torch.floor(torch.rand(tasks.shape[0], num_context, device=device) * counts.unsqueeze(1)).long()
		rows = self._ctx_rows[(starts.unsqueeze(1) + rand).view(-1)].long()
		storage = self._buffer._storage
		td_t = storage[rows]
		td_p = storage[rows-1]
		ctx = torch.cat([
			td_p.get('obs'),
			td_t.get('action'),
			td_t.get('reward').unsqueeze(-1),
			td_t.get('obs'),
		], dim=-1)
		ctx = ctx.view(tasks.shape[0], num_context, -1)
		return ctx.to(self._device, non_blocking=True)

	def sample(self):
		"""Sample a batch of subsequences from the buffer."""
		td = self._buffer.sample().view(-1, self.cfg.horizon+1).permute(1, 0)
		obs, action, reward, terminated, task = self._prepare_batch(td)
		ctx = self.sample_context(task) if (self.cfg.multitask and task is not None) else None
		return obs, action, reward, terminated, task, ctx

import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
import warnings
warnings.filterwarnings('ignore')

from glob import glob
from pathlib import Path
from time import time

import hydra
import torch
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.logger import Logger
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def find_checkpoints(checkpoint):
	"""
	Expand `checkpoint` into an ordered list of checkpoint files. Accepts a
	single .pt file, a directory of checkpoints (e.g. the `models/` dir written
	during training), or a glob pattern. Ordered by the training iteration in
	the filename, with non-numeric names (e.g. `final.pt`) last.
	"""
	path = Path(checkpoint)
	if path.is_file():
		fps = [path]
	elif path.is_dir():
		fps = [Path(fp) for fp in glob(str(path / '*.pt'))]
	else:
		fps = [Path(fp) for fp in glob(str(checkpoint))]
	assert len(fps) > 0, f'No checkpoints found at {checkpoint}'
	return sorted(fps, key=lambda fp: (0, int(fp.stem), '') if fp.stem.isdigit() else (1, 0, fp.stem))


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
	"""
	Script for evaluating multi-task TD-MPC2 checkpoints with the full set of
	offline-trainer evaluation metrics (per-task reward/success, normalized
	scores, gradient-conflict diagnostics), logged to wandb with the same
	metric names and `iteration` x-axis as training runs. Accepts a single
	checkpoint, a directory of checkpoints, or a glob pattern, and evaluates
	all of them in one run so full training curves can be reconstructed.

	Most relevant args:
		`task`: task set to evaluate on (mt30/mt80)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`checkpoint`: single .pt file, directory of checkpoints, or glob pattern
		`data_dir`: offline dataset dir; required for gradient-conflict metrics
		`eval_episodes`: number of episodes to evaluate on per task (default: 10)
		`exp_name`: use a distinct name to keep the wandb run separate from training

	See config.yaml for a full list of args.

	Example usage:
	```
		$ python evaluate_checkpoints.py task=mt30 model_size=5 checkpoint=/path/to/logs/mt30/1/default/models data_dir=/path/to/data
		$ python evaluate_checkpoints.py task=mt80 model_size=48 checkpoint='/path/to/models/*0000.pt' log_grad_conflict=false
		$ python evaluate_checkpoints.py task=mt30 model_size=5 checkpoint=/path/to/mt30-5M.pt wandb_project=my-project wandb_entity=me
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	cfg = parse_cfg(cfg)
	assert cfg.multitask, 'This script mirrors the offline trainer and only supports multitask (mt30/mt80) evaluation.'
	set_seed(cfg.seed)

	fps = find_checkpoints(cfg.checkpoint)
	print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
	print(colored(f'Found {len(fps)} checkpoint(s):', 'blue', attrs=['bold']))
	for fp in fps:
		print(colored(f'  {fp}', 'blue'))

	trainer = OfflineTrainer(
		cfg=cfg,
		env=make_env(cfg),
		agent=TDMPC2(cfg),
		buffer=None,
		logger=Logger(cfg),
	)

	# Gradient-conflict diagnostics sample batches from the offline dataset,
	# so they are only available when `data_dir` is provided.
	log_grad_conflict = cfg.get('log_grad_conflict', True)
	if log_grad_conflict and cfg.data_dir != '???':
		trainer._load_dataset()
	elif log_grad_conflict:
		log_grad_conflict = False
		print(colored('data_dir not set: skipping gradient-conflict metrics.', 'yellow', attrs=['bold']))

	start_time = time()
	for fp in fps:
		step = trainer.agent.load(fp)
		if not step and fp.stem.isdigit():
			step = int(fp.stem)  # older checkpoints do not store their iteration
		print(colored(f'\nEvaluating {fp} (iteration {step})', 'green', attrs=['bold']))
		metrics = {
			'iteration': step,
			'elapsed_time': time() - start_time,
		}
		metrics.update(trainer.eval())
		if log_grad_conflict:
			metrics.update(trainer.agent.grad_conflict(trainer.buffer))
		trainer.logger.pprint_multitask(metrics, cfg)
		trainer.logger.log(metrics, 'pretrain')
	trainer.logger.finish()


if __name__ == '__main__':
	evaluate()

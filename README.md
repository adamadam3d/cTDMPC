<h1>cTDMPC: Contextual TD-MPC2</h1>

TD-MPC2 with PEARL-style task inference — the learned task embedding is replaced by a probabilistic context encoder, turning multi-task TD-MPC2 into a meta-RL agent that infers the task from experience instead of being told its identity.

Built on the official implementation of [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://www.tdmpc2.com) by [Nicklas Hansen](https://nicklashansen.github.io), [Hao Su](https://cseweb.ucsd.edu/~haosu), and [Xiaolong Wang](https://xiaolonw.github.io) (UC San Diego), combined with the task-inference mechanism from [PEARL: Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables](https://arxiv.org/abs/1903.08254) (Rakelly et al., 2019).

[[TD-MPC2 Website]](https://www.tdmpc2.com) [[TD-MPC2 Paper]](https://arxiv.org/abs/2310.16828) [[PEARL Paper]](https://arxiv.org/abs/1903.08254) [[Dataset]](https://www.tdmpc2.com/dataset)

----

## What is different from TD-MPC2?

Vanilla multi-task TD-MPC2 conditions every component (encoder, dynamics, reward, policy, Q-functions) on a learned embedding `e` looked up by the ground-truth task ID. This fork removes that lookup entirely:

- **Context encoder (MLP + Product of Gaussians).** An MLP maps each context transition `(s, a, r, s')` to a Gaussian factor `(mu_i, sigma_i)`. Factors are combined into a posterior `q(z|c)` via a Product of Gaussians, with the `N(0, I)` prior included as a factor so that an empty context cleanly reduces to the prior.
- **Task inference, not task identity.** No network ever sees the task ID. All components are conditioned on a latent `z` sampled from `q(z|c)`. The ID survives only for env-side plumbing (action-space masks and per-task discounts) — the standard PEARL assumption that the environment is given while task semantics are hidden.
- **Training (offline, mt30/mt80).** For each batch element, `num_context` transitions of the same task are sampled independently from the offline dataset. The encoder is trained through all world-model losses (consistency, reward, value) plus a `KL(q(z|c) || N(0, I))` regularizer weighted by `kl_coef`. The latent is detached for the policy update.
- **Inference (incremental in-episode posterior).** At episode start the agent acts under the prior. After every step, the new transition is appended to an online context FIFO (`context_window`) and the posterior is recomputed — the agent figures out which task it is in *while acting*. Evaluation uses the posterior mean.

New config parameters (see `config.yaml`):

| argument | default | description |
| --- | --- | --- |
| `num_context` | 64 | context transitions per batch element during training |
| `context_window` | 100 | max online context transitions kept during rollout |
| `kl_coef` | 0.1 | weight of the KL regularizer (PEARL's `kl_lambda`) |

Single-task training and evaluation are completely unaffected — all changes are gated behind `multitask=true`.

**Important:** the official pretrained multi-task TD-MPC2 checkpoints are **not compatible** with this fork (`_task_emb` was removed and `_ctx_enc` added). Multi-task agents must be trained from scratch. Official *single-task* checkpoints remain compatible.

----

## Getting started

You will need a machine with a GPU and at least 12 GB of RAM for single-task online RL, and 128 GB of RAM for multi-task offline RL on the 80-task dataset. A GPU with at least 8 GB of memory is recommended for single-task online RL; larger multi-task models require correspondingly more memory.

A `Dockerfile` is provided for easy installation:

```
cd docker && docker build . -t <user>/tdmpc2:1.0.1
```

This docker image contains all dependencies needed for running DMControl. If you prefer `conda`:

```
conda env create -f docker/environment.yaml
```

The `docker/environment.yaml` file installs dependencies required for training on DMControl tasks. Other domains (Meta-World, ManiSkill2, MyoSuite) can be installed by following the instructions in `docker/environment.yaml`. For ManiSkill2 assets, MuJoCo licensing, and other domain-specific setup, refer to the [upstream TD-MPC2 instructions](https://github.com/nicklashansen/tdmpc2#getting-started).

----

## Supported tasks

The codebase supports all **104** continuous control tasks from **DMControl**, **Meta-World**, **ManiSkill2**, and **MyoSuite** used in the TD-MPC2 paper. See below table for expected name formatting for each task domain:

| domain | task
| --- | --- |
| dmcontrol | dog-run
| dmcontrol | cheetah-run-backwards
| metaworld | mw-assembly
| metaworld | mw-pick-place-wall
| maniskill | pick-cube
| maniskill | pick-ycb
| myosuite  | myo-key-turn
| myosuite  | myo-key-turn-hard

Multi-task (meta-RL) training and evaluation is specified by setting `task=mt80` or `task=mt30` for the 80-task and 30-task sets, respectively, and requires downloading the corresponding [offline dataset](https://www.tdmpc2.com/dataset) and setting `data_dir`. Use argument `obs=rgb` for image observations in DMControl tasks (single-task only).

----

## Example usage

### Training

```
$ python train.py task=mt80 model_size=48 batch_size=1024 data_dir=/path/to/mt80
$ python train.py task=mt30 model_size=5 batch_size=256 data_dir=/path/to/mt30
$ python train.py task=dog-run steps=7000000
$ python train.py task=walker-walk obs=rgb
```

Multi-task runs train the context encoder end-to-end with the world model; the `kl_loss` metric tracks the KL regularizer. We recommend configuring [Weights and Biases](https://wandb.ai) (`wandb`) in `config.yaml` to track training progress.

### Evaluation

```
$ python evaluate.py task=mt30 model_size=5 checkpoint=/path/to/your-mt30.pt
$ python evaluate.py task=dog-run checkpoint=/path/to/dog-1.pt save_video=true
```

During multi-task evaluation the agent starts each episode under the prior and adapts its task belief online from the transitions it observes. Remember that only checkpoints trained with this fork can be evaluated in multi-task mode.

### Tests

CPU unit checks for the context encoder (Product of Gaussians math, posterior inference, masking, per-task context sampling) can be run with:

```
$ python test_pearl_context.py
```

----

## Citation

If you build on this fork, please cite the original TD-MPC2 and PEARL papers:

```
@inproceedings{hansen2024tdmpc2,
  title={TD-MPC2: Scalable, Robust World Models for Continuous Control},
  author={Nicklas Hansen and Hao Su and Xiaolong Wang},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```
```
@inproceedings{rakelly2019pearl,
  title={Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables},
  author={Kate Rakelly and Aurick Zhou and Deirdre Quillen and Chelsea Finn and Sergey Levine},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2019}
}
```

----

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. Note that the repository relies on third-party code, which is subject to their respective licenses.

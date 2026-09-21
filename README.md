# LRC-JEPA

Latent Residual-Context JEPA learns a dynamic latent state `z` and a residual
context `u` from image-action sequences. Training combines latent prediction,
SIGReg on `z`, VICReg on `u`, and reconstruction from `z + u`. Planning uses
predicted dynamic latents; probes evaluate the learned representations.

This publication release contains training, model predictive control (MPC),
physical-property probes, and one pretrained checkpoint for each of Push-T,
TwoRoom, Reacher, Cube, and Bridge-v2.

**Commercial use of the original LRC-JEPA contributions and checkpoints requires
prior written permission.** See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Installation

Use Python 3.10 and a virtual environment. From this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-sim.txt
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. For training and offline
Bridge-v2 evaluation without simulators, install `requirements.txt` instead.
Choose PyTorch wheels appropriate for your CUDA driver or CPU before installing
the requirements. The validated local setup used PyTorch 2.12.0, torchvision
0.27.0, and Python 3.10.20; the CUDA wheels were built for CUDA 13.2.

Simulation rendering needs an OpenGL backend even when videos are disabled.
The runtime defaults to EGL on Linux and GLFW on Windows. Set `MUJOCO_GL` before
running if your platform needs another backend. Dependencies retain their own
licenses and are installed separately.

## Data

Set `LOCAL_DATASET_DIR` to your dataset root (the default is `./data`):

```bash
export LOCAL_DATASET_DIR=/path/to/datasets
```

PowerShell:

```powershell
$env:LOCAL_DATASET_DIR = 'C:/path/to/datasets'
```

The provided training configurations expect these prepared dataset directories:

| Config name | Dataset directory | Raw action dimensions |
| --- | --- | ---: |
| `pusht` | `pusht_expert_train.npy/` | 2 |
| `tworoom` | `tworoom.npy/` | 2 |
| `reacher` | `reacher.npy/` | 2 |
| `cube` | `cube_single_expert.npy/` | 5 |
| `bridge_v2` | `bridge_v2_clean.npy/` | 7 |

Despite the `.npy` suffix, each is a **directory of row-aligned arrays**, not a
single NumPy file. It contains `metadata.json`, `ep_len.npy`, `ep_offset.npy`,
and column arrays such as `pixels.npy` and `action.npy`. For sharded storage,
columns instead contain numbered files such as `pixels/000000.npy`.
`metadata.json` lists `columns` and `arrays`, whose entries include `shape` and
`dtype`; sharded datasets also set `storage: "npy_sharded"`, `shard_rows`, and
each column's `num_shards`. Episode lengths and offsets index the raw rows.

Training requires RGB `pixels` (uint8, 224 × 224) and floating-point `action`
columns. The released runs use four sampled frames, frameskip 5, three history
frames, and one prediction target. Actions within each five-step interval are
concatenated, giving model action widths 10, 25, or 35. Images are ImageNet
normalized; action normalization is fitted from the supplied dataset. Use the
same prepared dataset to reproduce checkpoint evaluation.

HDF5 datasets supported by `stable-worldmodel` can also be used, for example:

```bash
python train.py --config-name pusht data.dataset.name=pusht_expert_train.h5
```

Evaluation resolves dataset names to a prepared directory or HDF5 file under
`LOCAL_DATASET_DIR` (or its `datasets/` subdirectory). Simulator evaluation and
probes additionally need the relevant episode/state columns:

| Environment | Additional columns used by MPC and probes |
| --- | --- |
| Push-T | `episode_idx` (or `ep_idx`), `step_idx`, `proprio`, `state` |
| TwoRoom | episode/step indices, `proprio`, `pos_agent` |
| Reacher | episode/step indices, `qpos`, `qvel`, `finger_pos`, `target_pos` |
| Cube | episode/step indices, `qpos`, `qvel`, `privileged_block_0_pos`, `privileged_block_0_quat` |

Datasets, Bridge annotations, and data preparation tools are not bundled.
Obtain data separately under its original terms. Bridge checkpoint evaluation
uses the prepared, filtered `bridge_v2_clean.npy` dataset; a different conversion
or filtering will not necessarily reproduce the reported metrics.

## Training

```bash
python train.py --config-name pusht
python train.py --config-name tworoom
python train.py --config-name reacher
python train.py --config-name cube
python train.py --config-name bridge_v2
```

Each configuration provides the checkpoint's architecture, loss weights,
batch size, optimizer, and scheduler settings. The simulated runs use 10 epochs;
Bridge-v2 uses a 20-epoch schedule.
The release seeds model initialization and data splitting. A new training run
is not guaranteed to reproduce the original weights or scores exactly.

Example overrides and a short CPU smoke run:

```bash
python train.py --config-name cube loader.batch_size=64 num_workers=4
python train.py --config-name pusht trainer.accelerator=cpu trainer.devices=1 trainer.precision=32-true trainer.max_epochs=1 trainer.max_steps=1 +trainer.limit_train_batches=1 +trainer.limit_val_batches=0 loader.batch_size=2 num_workers=0
```

New runs write `config.yaml`, epoch model weights, and resumable Lightning
checkpoints to `outputs/<environment>/<date>/<time>/`. To resume, set
`hydra.run.dir` to that existing run directory; `checkpoints/last.ckpt` is loaded
when present. Published weights contain model parameters only and cannot restore
optimizer progress. No experiment-tracking account is required.

## MPC evaluation

First [download the pretrained checkpoints](#checkpoints) from Zenodo and place
them in the repository's `checkpoints/` directory.

```bash
python eval.py --config-name pusht
python eval.py --config-name tworoom
python eval.py --config-name reacher
python eval.py --config-name cube
```

Each command defaults to the corresponding `checkpoints/<environment>/` folder.
Override with `policy=/path/to/checkpoint_directory` or a weights file with an
adjacent `config.yaml`. Planning uses CEM with 300 candidates, 30 iterations,
and 30 elites. The provided protocol evaluates 50 start-goal pairs, with a
25-step goal offset and a 50-step environment budget.

For a quick functionality check (not a reproduction of paper results):

```bash
python eval.py --config-name pusht eval.num_eval=1 solver.num_samples=4 solver.topk=2 solver.n_steps=1 solver.device=cpu
```

Bridge-v2 uses **offline latent planning**, not closed-loop robot execution:

```bash
python eval.py --config-name bridge_v2
```

This ranks recorded expert actions against random alternatives and optimizes
actions in latent space. Recall and latent costs do not measure real-world
robot success. Text and JSON metrics are written under `results/`; change the
root with `output.directory=/path/to/results`. Evaluation never writes into the
published checkpoint directories and does not save videos.

## Probe evaluation

```bash
python eval_probe.py --config-name pusht output.filename=probe_results.txt
python eval_probe.py --config-name tworoom output.filename=probe_results.txt
python eval_probe.py --config-name reacher output.filename=probe_results.txt
python eval_probe.py --config-name cube output.filename=probe_results.txt
```

Simulated probes default to 10,000 sampled frames, an 80/20 frame split, linear
ridge regression and an MLP, reporting normalized MSE and mean Pearson
correlation. This is a frame split, not an episode-disjoint split. Push-T,
TwoRoom, and Reacher default to `z`; Cube evaluates `z`, `u`, and concatenated
`zu`, averaging `u` tokens. Override existing fields with `eval.<field>=...` and
add optional fields with `+eval.<field>=...`, for example:

```bash
python eval_probe.py --config-name pusht +eval.probe_feature_sets=[z,u,zu] +eval.probe_samples=10000 output.filename=probe_results.txt
```

Bridge-v2 requires object-property labels aligned to the prepared dataset:

```bash
python eval_probe.py --config-name bridge_v2 eval.probe_labels=/path/to/labels.csv
```

The CSV must contain `row_idx`, `episode_idx`, `frame_local_idx`, `gripper_xy`,
and at least one numbered `moving_N`, `static_N`, and `background_N` object slot.
Each slot has `_xy`, `_color_rgb`, and `_category` columns. Coordinates and RGB
values are JSON arrays in CSV cells. Foreground categories are `food`,
`cookware`, `towel`, and `small object`; background categories are `table`,
`countertop`, and `sink`. The evaluator selects the first complete slot in each
group and drops incomplete rows.

Alternatively supply an NPZ with `row_idx`, `episode_idx`, and target arrays,
optionally accompanied by `target_specs.json` describing each target's `kind`
(`classification` or `regression`) and `group`. Integer targets default to
classification and floating-point targets to regression. Bridge probes use an
episode-disjoint 80/20 split, train-only target normalization, and evaluate
`z`, `u`, and `zu`. Classification reports accuracy and macro F1.

## Checkpoints

**[Download the pretrained checkpoints from Zenodo][zenodo-checkpoints].**

Download the weights and accompanying configurations, then extract or place
them under `checkpoints/` in the repository root. Each environment must have
both `weights.pt` and `config.yaml`, for example:

```text
checkpoints/cube/weights.pt
checkpoints/cube/config.yaml
```

The released checkpoints and their reported metrics are listed below.
Each checkpoint directory contains `weights.pt` and a portable `config.yaml`.
The five weights total approximately 404 MiB. See
[checkpoints/manifest.json](checkpoints/manifest.json) for file sizes and
SHA-256 checksums. All five files are tensor state dictionaries loaded with
`torch.load(..., weights_only=True)` and strict parameter matching. They include
the encoder, predictor, context pooler, and reconstruction decoder.

### Simulated environments: MPC

We report the highest recorded MPC success rate for the best checkpoint in
each environment. Each evaluation uses 50 episodes.

| Environment | Checkpoint directory after download | Highest MPC success rate (%) ↑ |
| --- | --- | ---: |
| Push-T | `checkpoints/pusht/` | 96.0 |
| TwoRoom | `checkpoints/tworoom/` | 98.0 |
| Reacher | `checkpoints/reacher/` | 90.0 |
| Cube | `checkpoints/cube/` | 80.0 |

### Bridge-v2: offline latent planning

Offline metrics for the [Bridge-v2 checkpoint][zenodo-checkpoints], averaged
over 50 trajectories. Place its files in `checkpoints/bridge_v2/`.

| Metric | Value |
| --- | ---: |
| Expert recall@1 ↑ | 0.34 |
| Expert recall@5 ↑ | 0.64 |
| Expert recall@10 ↑ | 0.80 |
| Expert rank percentile ↓ | 0.020 |
| CEM / expert cost ratio ↓ | 0.806 |
| Expert vs. shuffled improvement ratio ↑ | 0.055 |

### Loading a checkpoint

From the repository root:

```python
from eval import load_local_policy_model

model, _, config = load_local_policy_model("checkpoints/cube")
model.eval()
```

Checkpoints are distributed through Zenodo; Git LFS is not required to download
or use them.

## Repository layout

```text
train.py              Training entry point and losses
eval.py               CEM/MPC and offline Bridge-v2 evaluation
eval_probe.py         Physical-property probes
jepa.py, module.py    Model and neural network components
utils.py              Dataset loading and checkpoint callbacks
runtime.py, compat.py Runtime and dependency compatibility
config/train/         Five training configurations
config/eval/          Five evaluation configurations and CEM settings
checkpoints/          Five model-only checkpoints and inference configs
```

## Validation

The condensed release was checked in the existing Python 3.10 environment:
strict loading and SHA-256 verification of all five checkpoints, exact
encoder/context/action/predictor output agreement with the original Bridge
model, one-batch training, one-episode MPC in each simulator, offline Bridge
planning, and small linear/MLP probe runs in all five environments. These are
functionality checks with reduced evaluation settings. Full training runs and
the reported benchmark scores were not rerun; installation in a fresh
environment has not been independently tested.

## Attribution and licensing

This work builds on the JEPA world-model codebase, `stable-pretraining`, and
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel). Please
cite the accompanying LRC-JEPA paper when using this release in research.

The original contributions and model parameters are available under the custom
[LRC-JEPA Noncommercial Research License](LICENSE). Commercial use, including
commercial products, services, and commercially directed model adaptation,
requires prior written permission from the relevant copyright holders. Direct
permission requests to the repository maintainers. This is a source-available
noncommercial release. Upstream MIT-covered code retains its original rights
and notice in [LICENSES/UPSTREAM-MIT.txt](LICENSES/UPSTREAM-MIT.txt).

[zenodo-checkpoints]: https://zenodo.org/records/22853413?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImNjMTMzYmJhLWQwYzAtNDA3NS05Mzk3LTVkMTU1ZjExYzA1OSIsImRhdGEiOnt9LCJyYW5kb20iOiI5Y2I3N2U2ODM4ZjdmMmZkYzY4YjI0ZmVlZDk1NWY3ZCJ9.GPGLc6S_OkLz5Jz0HhWLA0f6gawoDAfZdJECvRh4hoOWdLdxLlcJaXSL5oH7TGlYgIXACKAIAEXHtT0cFA6jbw

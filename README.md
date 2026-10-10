# CDA-UNet

**A Conditional Dynamic Adapter U-Net for Automated Cytometric Gating**

CDA-UNet uses one conditional segmentation model for all gates within a cytometry
panel. It converts each pair of measurement channels into a density image,
predicts a binary gate, and maps that prediction back to the original cells.
The model is conditioned on the **ordered X-channel identity, Y-channel identity,
and gate identity**. These conditions are channel labels, not cell coordinates.

This repository contains data preparation, the model, cross-validation training,
hierarchical inference, and cell-level evaluation for **mass cytometry (9 gates)**
and **flow cytometry (7 gates)**. Train the two modalities separately.
No subject data, prediction files, or trained weights are distributed.

## Standard configuration

Only the standard mass and flow configurations are included.

| Setting | Value |
|---|---|
| Density-image size | 101 × 101 |
| Conditions | X-channel ID + Y-channel ID + gate ID |
| Dynamic adapter width (`rank`) | 32 |
| U-Net encoder widths | 64, 128, 256, 512; bottleneck 1024 |
| Normalization | Input-domain BN in encoder; per-gate BN in decoder |
| Optimizer / learning rate | Adam / 0.0001 |
| Loss / batch size | Binary cross-entropy with logits / 16 |
| Training duration | 1000 epochs; final checkpoint |
| Cross-validation | 8 folds (mass); 10 folds (flow) |
| Training parent populations | Manual consensus, training samples only |
| Default test parent populations | Cells retained by this model's preceding gates |

Both released configurations specify **1000 epochs**. Run the evaluation commands
to obtain scores for the trained checkpoints; the manuscript reports its own
experimental runs. See [data preparation and evaluation](docs/reproduction.md)
for the population selection rules and metric definitions.

## Installation

Use Python 3.10 or newer. A CUDA-capable PyTorch installation is recommended for
full-size training; CPU execution is supported for inference and small checks.
Select a PyTorch build appropriate for your environment using the
[official installation instructions](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/Intelligent-Medical-Research-Group/CDA-UNet.git
cd CDA-UNet
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
cda-unet --help
```

`python -m cda_unet` is equivalent to `cda-unet` in every command below.

## Data

Download the original annotated cell-level CSV files from
[Pennsieve Discover, dataset 408, version 2](https://discover.pennsieve.io/datasets/408/version/2).
Use the dataset's published terms and cite the source study. Keep each sample in
one file and preserve event order. The loader expects the released, transformed
CSV measurements; it does not ingest raw FCS or perform compensation.

```text
data/
├── mass/
│   ├── subject_001.csv
│   └── ...
└── flow/
    ├── subject_001.csv
    └── ...
```

CSV discovery is recursive. Each filename stem must be unique within its
modality, and the input directory must contain measurement CSVs only. One file
is one split unit: keep related measurements from the same subject together
before preparing data if extending the supplied dataset.

Required measurement columns:

| Panel | Exact CSV column names |
|---|---|
| Mass | `Event_length`, `CD45`, `CD19`, `CD16`, `CD294`, `CD3`, `CD66b`, `DNA1`, `DNA2` |
| Flow | `FSC-A`, `FSC-W`, `SSC-A`, `SSC-W`, `LIVEDEAD`, `CD3Q605`, `CD45RAQ655`, `CD8Q705`, `CD4PETR`, `CCR7PECY7` |

Preparation and evaluation also require `GaterConsensus`, containing the deepest
attained population. Ancestor labels are expanded using
[`schema.py`](src/cda_unet/schema.py). Prediction requires measurement columns
only and ignores annotation columns even when present. Missing measurements,
non-finite values, unknown training labels, and inconsistent supplied flow gate
labels cause explicit errors.

### Prepare density images and splits

```bash
cda-unet prepare --config configs/mass.yaml --data-dir data/mass --cache-dir cache/mass
cda-unet prepare --config configs/flow.yaml --data-dir data/flow --cache-dir cache/flow
```

Preparation creates a compact image/target cache and `manifest.json`, recording
the channel schema, sample hashes, and train/test membership. Splits are fixed
before training. Reusing a cache verifies the inputs; changed data or split
settings require a new cache directory.

For each gate, parent events define the coordinate range. Each axis is scaled
to [0, 1], multiplied by 100, and rounded to the nearest integer: **101 positions
from 0 through 100**, not 101 histogram intervals. Counts are divided by their
maximum, and positive training cells define a convex-hull target. See
[`data.py`](src/cda_unet/data.py) for empty/degenerate cases and backprojection.

## Training

Train all folds sequentially on one device:

```bash
cda-unet train --config configs/mass.yaml --cache-dir cache/mass --output-dir runs/mass --fold all --device cuda:0
cda-unet train --config configs/flow.yaml --cache-dir cache/flow --output-dir runs/flow --fold all --device cuda:0
```

The default is exactly 1000 epochs. Every batch contains one gate, so the correct
BN domain is updated. Subjects and gate batches are shuffled each epoch. CUDA
training uses bfloat16 when supported; otherwise training uses float32. Test
images never enter the optimizer pipeline, and no test score selects a checkpoint.

Outputs for each fold include:

```text
runs/mass/fold_00/
├── config.json
├── history.json
├── epoch_0100.pt       # periodic checkpoints every 100 epochs
├── ...
├── epoch_1000.pt
└── last.pt            # latest saved checkpoint, including optimizer state
```

Resume an interrupted fold from its latest saved checkpoint:

```bash
cda-unet train --config configs/mass.yaml --cache-dir cache/mass --output-dir runs/mass --fold 0 --device cuda:0 --resume runs/mass/fold_00/last.pt
```

Checkpoints include model and optimizer states, completed epoch, configuration,
random-number state, cache identity, and split membership. Existing runs require
an explicit `--resume`; a different cache, fold, or configuration is rejected.
`--epochs N` overrides the total target epoch count for a local smoke check or
continuation; it does not mean N additional epochs. Report the actual completed
epoch stored in the checkpoint.

### Use two GPUs

Run independent folds on the two devices. Prepare the cache once first. For mass
cytometry, use these two terminals (change the modality and fold lists for flow):

```bash
# Terminal 1: GPU 0
for fold in 0 2 4 6; do
  cda-unet train --config configs/mass.yaml --cache-dir cache/mass --output-dir runs/mass --fold "$fold" --device cuda:0 || break
done
```

```bash
# Terminal 2: GPU 1
for fold in 1 3 5 7; do
  cda-unet train --config configs/mass.yaml --cache-dir cache/mass --output-dir runs/mass --fold "$fold" --device cuda:1 || break
done
```

For flow cytometry use folds `0 2 4 6 8` and `1 3 5 7 9`, `configs/flow.yaml`,
`cache/flow`, and `runs/flow`. Each process trains a complete model on one GPU;
this is fold parallelism, not distributed training of a single fold.

## Evaluation

Evaluate the held-out samples recorded in one checkpoint:

```bash
cda-unet evaluate --checkpoint runs/mass/fold_00/last.pt --data-dir data/mass --output-dir outputs/example_mass_fold_00 --device cuda:0
```

After all folds are trained, evaluate and aggregate them:

```bash
for checkpoint in runs/mass/fold_*/last.pt; do
  fold=$(basename "$(dirname "$checkpoint")")
  cda-unet evaluate --checkpoint "$checkpoint" --data-dir data/mass --output-dir "outputs/mass/$fold" --device cuda:0 || break
done
cda-unet aggregate --reports outputs/mass/fold_*/report.json --output-dir outputs/mass/overall
```

Use a new output directory when repeating an evaluation. Repeat with `flow`
for the other modality.

`report.json` retains full-precision metrics, confusion counts, actual epoch,
parent policy, and provenance. CSV summaries display four decimal places.
Scores are calculated for each sample and gate **over all original events**,
including events rejected by a parent. Per-gate scores average samples equally;
`Overall` averages gates equally. Undefined precision, recall, or F1 is zero.
The aggregator rejects overlapping test samples and incompatible reports.
Partial aggregation is allowed and marked `complete_cross_validation: false`;
report full cross-validation only after all configured folds are included.

### Predict a new sample

```bash
cda-unet predict --checkpoint runs/mass/fold_00/last.pt --input new_sample.csv --output outputs/new_sample.npz --device cuda:0
```

The NPZ contains `predictions` (events × gates, binary), `gate_names`,
`sample_id`, and `source_sha256`. Rows retain their original CSV order. For
example:

```python
import numpy as np

with np.load("outputs/new_sample.npz", allow_pickle=False) as result:
    predictions = result["predictions"]
    gate_names = result["gate_names"].tolist()
```

By default, gates are inferred in hierarchy order. Empty parents produce empty
children; children cannot include events outside their predicted parent.
Sibling gates remain separate binary predictions and may overlap.

### Using saved upstream predictions

To use saved upstream predictions, pass `--parents-dir PATH` to `evaluate`
or `--parents-file FILE.npz` to `predict`. This explicitly changes the recorded
parent policy to `provided`. For training with supplied parents, set
`training.parent_policy: provided` in a local copy of the standard YAML and
pass `--parents-dir PATH` to `prepare` using a new cache directory.

Each sample's NPZ must contain `predictions` (binary events × gates),
`gate_names` in the schema order, and a scalar `source_sha256` matching the
exact CSV. The `predict` command produces this format. The models generating
these predictions must respect the outer train/test split; the loader checks identity
and shape, but cannot establish how an external model was trained.

## Architecture

Four encoder stages and a bottleneck share convolutional weights across gates.
Each decoder stage has a conditional residual adapter: a 1 × 1 down-projection,
a generated 3 × 3 convolution, group normalization, generated affine terms,
and a zero-initialized 1 × 1 up-projection. A shared binary head produces logits.

The condition encoder combines ordered channel embeddings and a gate embedding.
Its stage-specific generators create adapter kernels and affine coefficients.
BN is routed by `(parent, X channel, Y channel)` in the encoder/bottleneck
(5 mass domains; 6 flow domains), and by gate in the decoder (9 / 7 domains).
This is the manuscript's normalization scheme, sometimes abbreviated as
"per-gate BN"; sibling gates with the same input share encoder BN.

| Component | Source |
|---|---|
| Channel vocabulary and gate hierarchy | [`schema.py`](src/cda_unet/schema.py) |
| CSV loader, rasterization, target masks, backprojection | [`data.py`](src/cda_unet/data.py) |
| Conditional U-Net, adapters, routed BN | [`model.py`](src/cda_unet/model.py) |
| Preparation, training, checkpoints, evaluation | [`engine.py`](src/cda_unet/engine.py) |
| Sample/gate metrics and Overall averaging | [`metrics.py`](src/cda_unet/metrics.py) |
| Command-line interface | [`cli.py`](src/cda_unet/cli.py) |

## Tests

```bash
python -m pytest -q
```

Tests use invented measurements and CPU models with reduced backbone widths.
They check channel/label handling, rasterization, adapter routing, subject-level
splits, training and resume, annotation-free inference, hierarchy containment,
metric aggregation, and cache integrity. Full architecture parameter counts are
also checked. These are implementation checks, not a rerun of the paper's full
training experiments.

## Citation and licensing

The CDA-UNet manuscript is titled **CDA-UNet: A Conditional Dynamic Adapter U-Net
for Automated Cytometric Gating**. Publication details will be added when
available. Until then, cite this software repository and its commit identifier.

The bivariate density-image gating formulation and public dataset are described
in Chen et al. (2025), *Automated cytometric gating with human-level performance
using bivariate segmentation*, Nature Communications, 16, 1576.
[DOI: 10.1038/s41467-025-56622-2](https://doi.org/10.1038/s41467-025-56622-2).

Newly authored repository code and documentation are distributed under the
[MIT license](LICENSE). The original UNITO application is not vendored or
relicensed. Dataset and third-party software terms remain separate; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

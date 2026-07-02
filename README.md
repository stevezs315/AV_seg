# Improve Retinal Artery/Vein Classification via Channel Coupling

[![ESWA](https://img.shields.io/badge/ESWA-Paper-blue?logo=elsevier)](https://doi.org/10.1016/j.eswa.2025.130795)
[![GitHub](https://img.shields.io/badge/GitHub-Code-black?logo=github)](https://github.com/stevezs315/AV_seg)
[![Homepage](https://img.shields.io/badge/Homepage-Website-green?logo=googlechrome)](https://stevezs315.github.io/)

**Shuang Zeng, Chee Hong Lee, Kaiwen Li, Boxu Xie, Ourui Fu, Hangzhou He, Lei Zhu, Yanye Lu, Fangxiao Cheng**

*MILab, Department of Biomedical Engineering, Peking University*

*Corresponding Author: yanye.lu@pku.edu.cn*

---

## Introduction

Retinal artery/vein (A/V) classification is important for quantitative fundus-image analysis, but it is challenging because arteries and veins are thin, locally ambiguous, and tightly coupled with the global vessel tree.

We propose a channel-coupled A/V segmentation framework for retinal fundus images. The model predicts artery, vein, and vessel-tree channels jointly, and uses channel coupling to encourage structural consistency between the A/V branches and the complete vessel map.

-**Three-channel A/V formulation**: predicts artery, vein, and vessel-tree channels (`A`, `V`, `VT`) together.

-**Channel-coupling loss**: enforces consistency between artery/vein predictions and vessel-tree prediction.

-**Superpixel contrastive regularization**: introduces intra-image pixel-level contrastive learning guided by superpixel regions.

-**Recursive refinement**: supports RRWNet-style iterative refinement for vessel topology and A/V consistency.

-**Evaluation toolkit**: includes A/V, binary-vessel, topology-oriented, and significance-test utilities.

> **TL;DR:** Our method jointly models arteries, veins, and the full vessel tree, using channel coupling and superpixel-guided contrastive learning to improve A/V consistency and preserve fine retinal vessel structures.

---

## News

-**[2026-04-05]** Paper published in *Expert Systems With Applications*, Volume 305, Article 130795.

-**[2025-12-22]** DOI available: [10.1016/j.eswa.2025.130795](https://doi.org/10.1016/j.eswa.2025.130795).

---

## Method Overview

![Method Overview](assets/AV_seg.png)

The framework uses a three-channel output representation:

1.**Artery channel**: predicts artery pixels.

2.**Vein channel**: predicts vein pixels.

3.**Vessel-tree channel**: predicts the complete vessel structure.

The main training objective combines:

```bash

Total Loss = Basic BCE Loss + Channel-Coupled Vessel Consistency Loss + Intra-image Pixel-level Contrastive Loss
```

**Key modules:**

- **Channel-Coupled Vessel Consistency Loss**: A novel loss that enforces anatomical coherence among artery, vein, and vessel predictions by fusing them through min‑operations based on ground‑truth class labels, ensuring that arterial and venous predictions remain consistent with the overall vascular structure.
- **Intra-image Pixel-level Contrastive Loss**:   A regularization term that leverages superpixel segmentation to guide unsupervised contrastive learning, treating pixels within the same superpixel cluster as positive pairs and those from different clusters as negative pairs, thereby promoting more discriminative fine‑grained feature representations.

---

## Installation

```bash
conda create -n av_seg python=3.8.16
conda activate av_seg
pip install -r requirements.txt
```

---

## Model Weights and Predictions

Pre-trained model weights: "AV_Seg/model_pth"

| Model                 | train/test Dataset | Backbone          |
| ----------------------| -------------------| ------------------|
| RITE_generator_best   | RITE               | RRWNet + Our Loss |
| LES-AV_generator_best | LES-AV             | RRWNet + Our Loss |
| HRF_generator_best    | HRF                | RRWNet + Our Loss |

---

**The final predictions of RRWNet with our proposed loss can be found at "AV_Seg/predictions"**

## Dataset Preparation


All fundus datasets provide color retinal images with A/V annotations. Labels are stored as three-channel maps (`A`, `V`, `VT`), where artery and vein are subsets of the complete vessel tree.

| Dataset | Description | Training resolution | Images | 
| ------- | ----------- | ------ | ------ |
| RITE / DRIVE A/V | A/V benchmark derived from DRIVE | Full resolution (565 × 584)| 40 (20 train / 20 test) |
| LES-AV | A/V benchmark with glaucoma cases | Width resized to 576 px | 22 （11 train / 11 test） |
| HRF | High-resolution fundus with A/V labels | Width resized to 1024 px | 45 (30 train / 15 test)  |
| Fundus-AVSeg | Multi-disease fundus A/V segmentation | Resized to 1280 × 1280| 100 (80 train / 20 test) |

- RITE, [link](https://drive.google.com/file/d/154XKN2umLFXaghhXW--rIgDNzNRa4bxB/view?usp=drive_link)
- LES-AV, [link](https://drive.google.com/file/d/1MHK11qGJTDEpmvieJM0T0Q-799Iwvwjh/view?usp=drive_link)
- HRF, [link](https://drive.google.com/file/d/1JPsfHkk4Li_9EeXbJ5ZKUo7Y9qpYIrj7/view?usp=drive_link)
- Fundus-AVSeg, [link](https://drive.google.com/file/d/1sMJMa4XSMqta8LWMOISPv-J_DY7l76lo/view?usp=drive_link)

---

### Pre-Processing

All fundus images undergo offline preprocessing before training, including **global contrast enhancement** and **local intensity normalization**, following [Morano et al., 2021](https://doi.org/10.1016/j.media.2021.103274).

If field-of-view masks are unavailable, generate them with:

```bash
python generate_mask.py
```

Enhance fundus images with:

```bash
python preprocessing.py \
  --images-path train/_Data/<dataset>/training/images \
  --masks-path train/_Data/<dataset>/training/masks \
  --save-path train/_Data/<dataset>/training/enhanced
```

---

## Training

Run training from the `train/` directory. A helper script is provided at [train/1_train.sh](train/1_train.sh) with pre-configured commands for all datasets.

```bash
cd train

# Basic training with BCE3 loss
python3 train.py \
  --dataset RITE-train \
  --version RITE_rrwnet_bce3 \
  --model RRWNet \
  --gpu_id 0

# Training with min-max consistency loss
python3 train.py \
  --dataset RITE-train \
  --version RITE_rrwnet_minmax \
  --model RRWNet \
  --gpu_id 0 \
  --base_criterion BCE3wminmaxLoss

# Training with min-max consistency + superpixel contrastive loss (recommended)
python3 train.py \
  --dataset RITE-train \
  --version RITE_RRWNet_coupled_spcl \
  --model RRWNet \
  --gpu_id 0 \
  --base_criterion BCE3wminmaxLoss \
  --add_criterion SpCLLoss
```

Key arguments:

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `--dataset` | str | `RITE-train` | Dataset name: `RITE-train`, `LES-AV`, or `HRF-Karlsson-w1024` |
| `--model` | str | `RRWNet` | Network backbone (`RRWNet`) |
| `--base_criterion` | str | `BCE3Loss` | Base segmentation loss: `BCE3Loss` or `BCE3wminmaxLoss` |
| `--add_criterion` | str | `None` | Auxiliary regularization loss: `SpCLLoss` |
| `--num_iterations` | int | `5` | Number of recursive refinement iterations |
| `--version` | str | `""` | Experiment name (output saved to `train/__training/<version>/`) |
| `--gpu_id` | int | `0` | GPU device ID |
| `--learning_rate` | float | `1e-04` | Learning rate |
| `--num_epochs` | int | `None` | Number of training epochs (dataset-specific default if `None`) |
| `--base_channels` | int | `64` | Base channel count for the backbone |
| `--num_folds` | int | `4` | Number of cross-validation folds |
| `--data_folder` | str | `./_Data/` | Root directory for training data |
| `--use_checkpoint` | int | `0` | Set to `1` to resume from a checkpoint (`--checkpoint_path` and `--scheduler_path` required) |

Training outputs are saved to:

```text
train/__training/<version>/<dataset>/
├── generator_best.pth       # Best model checkpoint (use this for inference)
├── checkpoint_*.pth         # Intermediate checkpoints
├── config.json              # Run configuration (required for inference)
└── ...
```

The recommended checkpoint for inference is `generator_best.pth`.

---

## Inference

Run inference from the `train/` directory. A helper script is provided at [train/get_pred.sh](train/get_pred.sh).

```bash
cd train

python3 test.py \
  -w __training/RITE_RRWNet_minmax_0.19_spCL_0.05 \
  -i _Data/RITE/test/enhanced \
  -m _Data/RITE/test/enhanced_masks \
  -t RITE-test \
  --save_root ../eval/_Evaluation_Data \
  --save_path RITE_RRWNet_coupled_spcl
```

Key arguments:

| Argument | Description |
| -------- | ----------- |
| `-w`, `--weights` | Path to training output directory containing `generator_best.pth` and `config.json` |
| `-i`, `--images_path` | Path to the enhanced test images |
| `-m`, `--masks_path` | Path to the test FOV masks |
| `-t`, `--test_name` | Test set name (e.g., `RITE-test`, `LES-AV`, `HRF`); auto-detected if omitted |
| `--save_root` | Root directory for saving predictions (default: `../eval/_Evaluation_Data`) |
| `--save_path` | Sub-folder name under `<save_root>/<test_name>/` |

Predictions are saved to:

```text
eval/_Evaluation_Data/<test_name>/<save_path>/
```

---

## Evaluation

### Arbitrary Prediction Evaluation

For any prediction, ground-truth, and mask folders:

```bash
cd eval

python3 evaluate.py \
  -d RITE-test \
  -p _Evaluation_Data/RITE-test/RITE_RRWNet_coupled_spcl \
  -g _Evaluation_Data/RITE-test/gt_hu \
  -m _Evaluation_Data/RITE-test/masks \
  -t mav \
  --pixels both
```

### Dataset-Specific Comparison

```bash
cd eval

python3 compare_results.py \
  -d RITE-test \
  -p _Evaluation_Data/RITE-test/RITE_RRWNet_coupled_spcl \
  -v RITE_RRWNet_coupled_spcl
```

### Evaluation Modes and Options

| Argument | Description |
| -------- | ----------- |
| `-d`, `--dataset` | Dataset name (`RITE-test`, `LES-AV`, `HRF`, etc.) |
| `-p`, `--pred_path` / `--predictions_path` | Path to predicted images |
| `-g`, `--gt_path` | Path to ground-truth images |
| `-m`, `--mask_path` | Path to FOV mask images |
| `-t`, `--test` | Evaluation mode: `mav` (A/V + vessel-tree metrics) or `topo_mp` (topology metrics) |
| `--pixels` | Pixel scope: `both` (vessel + intersection), `intersection` (A/V crossings only), `all` (all vessel pixels) |
| `-v`, `--version` | Experiment version name for result tracking |
| `-n`, `--n_paths` | Number of centerline paths for topology analysis |

JSON results are saved to `eval/__results/`. Additional utilities include [eval/significance_test.py](eval/significance_test.py) for bootstrap-based statistical testing and [eval/efficiency_analysis.py](eval/efficiency_analysis.py) for model efficiency benchmarking.

---

## Project Structure

```text
AV_seg/
├── train/                         # Training, inference, models, losses, data loaders
│   ├── train.py                   # Main training entry point
│   ├── test.py                    # Prediction / inference entry point
│   ├── config.py                  # Dataset and training configuration
│   ├── r2av.py                    # Training orchestrator (R2Vessels class)
│   ├── factories.py               # Model and loss factories
│   ├── models.py                  # RRWNet model definition
│   ├── losses.py                  # BCE3, channel-coupling, SpCL, and recursive losses
│   ├── data_vessels.py            # VesselsDataset for loading images, GT, and masks
│   ├── data_utils.py              # Samplers, fold splitting, CSV utilities
│   ├── transformations.py         # Data augmentation transforms
│   ├── utils_pytorch.py           # UniversalFactory, early-stopping scheduler, checkpoint I/O
│   ├── 1_train.sh                 # Example training commands
│   └── get_pred.sh                # Example inference commands
├── eval/                          # Metric computation and evaluation utilities
│   ├── evaluate.py                # Arbitrary prediction / GT / mask folder evaluation
│   ├── compare_results.py         # Dataset-specific multi-model comparison
│   ├── constants.py               # Dataset definitions for evaluation
│   ├── test_utils.py              # Core metrics (classification, AUC, bootstrap, HD95)
│   ├── topo_metric.py             # Topology-oriented centerline metrics
│   ├── significance_test.py       # Bootstrap-based statistical significance testing
│   ├── efficiency_analysis.py     # Model efficiency / FLOPs benchmarking
│   └── net_utils.py               # Shared network utility modules
├── data_processing/               # Dataset conversion and distribution analysis
│   ├── data_distribution.py       # Pixel-level class distribution analysis
│   └── groundtruth.py             # Three-channel A/V/VT ground-truth generation
├── predictions/                   # Pre-computed predictions (RITE, LES-AV, HRF)
├── assets/                        # Figures and illustrations
├── preprocessing.py               # Fundus image enhancement (contrast + normalization)
├── generate_mask.py               # Field-of-view mask generation
├── utils.py                       # UNet padding and tensor conversion helpers
├── 1_preprocessing.sh             # Preprocessing script examples
├── requirements.txt
└── README.md
```

---

## Citation

If you find this repository useful in your research, please cite our paper:

```bibtex
@article{zeng2026channelcoupling,
  title   = {Improve retinal artery/vein classification via channel coupling},
  author  = {Zeng, Shuang and Lee, Chee Hong and Li, Kaiwen and Xie, Boxu and Fu, Ourui and He, Hangzhou and Zhu, Lei and Lu, Yanye and Cheng, Fangxiao},
  journal = {Expert Systems With Applications},
  volume  = {305},
  pages   = {130795},
  year    = {2026},
  doi     = {10.1016/j.eswa.2025.130795}
}
```

If you use the recursive refinement architecture or its evaluation protocol, please also cite the original [RRWNet](https://github.com/stevezs315/AV_seg) work.

---

## Contact

- **Shuang Zeng** (First Author): [stevezs@pku.edu.cn](mailto:stevezs@pku.edu.cn)
- **Yanye Lu** (Corresponding Author): [yanye.lu@pku.edu.cn](mailto:yanye.lu@pku.edu.cn)

For questions and issues, please open a [GitHub Issue](https://github.com/stevezs315/AV_seg/issues).

---

## Acknowledgement

This codebase builds upon [RRWNet](https://github.com/stevezs315/AV_seg). We thank the authors for their excellent work and for releasing their code.

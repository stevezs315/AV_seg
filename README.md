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

![Method Overview](AV_seg.png)

The framework uses a three-channel output representation:

1.**Artery channel**: predicts artery pixels.

2.**Vein channel**: predicts vein pixels.

3.**Vessel-tree channel**: predicts the complete vessel structure.

The main training objective combines:

```text

Total Loss = Base A/V Loss + Channel-Coupling Loss + Superpixel Contrastive Loss + Optional Recursive Loss
```

**Key modules:**

-**`BCE3Loss`**: base three-channel A/V/VT supervision.

-**`BCE3wminmaxLoss`**: channel-coupled variant used to align A/V predictions with the vessel-tree channel.

-**`SpCLLoss`**: superpixel-guided contrastive loss for intra-image pixel-level consistency.

-**Recursive refinement losses**: optional regularizers such as `SVoxelLoss`, `D25Loss`, and `topoLoss`.

Supported backbones include `RRWNet`, `RRWNetAll`, `RRUNet`, `WNet`, `UNet`, `UNet_pp`, `LadderNetv6`, `Rolling_Unet_M`, `AttU_Net`, `Iternet`, and `RSFConvUnet`.

---

## Installation

```bash

condacreate-nav_segpython=3.8.16

condaactivateav_seg

pipinstall-rrequirements.txt
```

The checked-in `requirements.txt` records the environment used by historical experiments. For a clean public release, it is recommended to create a smaller `requirements-minimal.txt` or `environment.yml`.

---

## Model Weights

Pre-trained weights are not bundled with the public repository by default because checkpoint files can be large. Example local checkpoints follow this layout:

| Model | Dataset | Checkpoint |

| ----- | ------- | ---------- |

| RRWNet + channel coupling + SpCL | RITE | `train/__training/RITE_RRWNet_minmax_0.19_spCL_0.05/generator_best.pth` |

| RRWNet baseline | RITE | `train/__training/RITE_RRWNet_BCE_base/generator_best.pth` |

For release, place large model weights in GitHub Releases, Google Drive, Zenodo, or Git LFS, then document the download path here.

---

## Dataset Preparation

### Supported Datasets

| Dataset | Description | Code name |

| ------- | ----------- | --------- |

| RITE / DRIVE A/V | Retinal artery/vein benchmark | `RITE-train` |

| LES-AV | Retinal artery/vein benchmark | `LES-AV` |

| HRF | High-resolution fundus A/V labels | `HRF-Karlsson-w1024` |

| Fundus-AVSeg | Fundus A/V segmentation data | `Fundus-AVSeg` |

| GAVE | Local GAVE-style training data | `GAVE` |

| Private large-field data | Internal/private data | `LF_private` |

The paper experiments are centered on four public datasets: **RITE**, **LES-AV**, **HRF**, and **Fundus-AVSeg**.

### Expected Layout

Training data are expected under `train/_Data/`:

```text

train/_Data/<dataset>/<split>/

├── images/ or enhanced/        # input fundus images

├── masks/ or enhanced_masks/   # field-of-view masks

└── av3/                        # 3-channel A/V/VT labels
```

For `Fundus-AVSeg`, the code expects:

```text

train/_Data/Fundus-AVSeg/

├── images/

├── masks/

├── annotation/

├── training.txt

└── testing.txt
```

### Pre-Processing

If field-of-view masks are unavailable, generate them with:

```bash

pythongenerate_mask.py
```

Enhance fundus images with:

```bash

pythonpreprocessing.py\

  --images-path train/_Data/<dataset>/training/images \

--masks-pathtrain/_Data/<dataset>/training/masks\

  --save-path train/_Data/<dataset>/training/enhanced
```

For a new dataset:

1. Split images into train/test sets.
2. Prepare masks and three-channel A/V/VT labels.
3. Run image enhancement if needed.
4. Register the dataset in `train/config.py`.

---

## Training

Run training from the `train/` directory:

```bash

cdtrain

python3train.py\

  --dataset RITE-train \

--versionRITE_RRWNet_coupled_spcl\

  --model RRWNet \

--gpu_id0\

  --base_criterion BCE3wminmaxLoss \

--add_criterionSpCLLoss
```

Useful arguments:

| Argument | Description |

| -------- | ----------- |

| `--dataset` | Dataset name, such as `RITE-train`, `LES-AV`, `HRF-Karlsson-w1024`, or `Fundus-AVSeg` |

| `--model` | Network backbone, such as `RRWNet`, `UNet`, `UNet_pp`, `Rolling_Unet_M`, or `RSFConvUnet` |

| `--base_criterion` | Base loss, usually `BCE3Loss` or `BCE3wminmaxLoss` |

| `--add_criterion` | Additional loss, for example `SpCLLoss` |

| `--recursive_criterion` | Optional recursive regularizer, such as `BCE3Loss`, `SVoxelLoss`, `D25Loss`, or `topoLoss` |

| `--num_iterations` | Number of recursive refinement iterations |

| `--version` | Experiment folder name under `train/__training/` |

Training outputs are saved to:

```text

train/__training/<version>/
```

The most useful inference checkpoint is usually `generator_best.pth`. `checkpoint_final.pth` indicates a completed run.

---

## Inference

Run inference from the `train/` directory and save predictions under `eval/_Evaluation_Data/`:

```bash

cdtrain

python3test.py\

  --weights __training/RITE_RRWNet_minmax_0.19_spCL_0.05 \

--images_path_Data/RITE/test/enhanced\

  --masks_path _Data/RITE/test/enhanced_masks \

--test_nameRITE-test\

  --save_root ../eval/_Evaluation_Data \

--save_pathRITE_RRWNet_coupled_spcl
```

The prediction folder will be:

```text

eval/_Evaluation_Data/RITE-test/RITE_RRWNet_coupled_spcl/
```

---

## Evaluation

For arbitrary prediction, ground-truth, and mask folders:

```bash

cdeval

pythonevaluate.py\

  -d RITE-test \

-p_Evaluation_Data/RITE-test/RITE_RRWNet_coupled_spcl\

  -g _Evaluation_Data/RITE-test/gt_hu \

-m_Evaluation_Data/RITE-test/masks\

  -t mav \

--pixelsboth
```

For dataset-specific comparison:

```bash

cdeval

pythoncompare_results.py\

  -d RITE-test \

-p_Evaluation_Data/RITE-test/RITE_RRWNet_coupled_spcl\

  -v RITE_RRWNet_coupled_spcl
```

Main evaluation modes:

| Mode | Description |

| ---- | ----------- |

| `mav` | Artery/vein and vessel-tree classification metrics |

| `topo_mp` | Topology-oriented metrics |

| `--pixels both` | Evaluate both full-vessel and intersection-pixel settings |

| `--pixels intersection` | Evaluate only the artery/vein intersection region |

| `--pixels all` | Evaluate all vessel pixels |

JSON results are written to `eval/__results/`. The helper scripts `eval/resultsxls.py` and `eval/topo_resultsxls.py` convert selected JSON outputs to Excel files.

---

## Project Structure

```text

AV_seg/

├── train/                         # Training, inference, models, losses, data loaders

│   ├── train.py                   # Main training entry point

│   ├── test.py                    # Prediction / inference entry point

│   ├── config.py                  # Dataset and training configuration

│   ├── factories.py               # Model and loss factories

│   ├── models.py                  # RRWNet / UNet-style model definitions

│   ├── losses.py                  # BCE3, channel-coupling, SpCL, topology losses

│   ├── networks/                  # Baseline network implementations

│   └── supervoxel_loss/           # Supervoxel / topology-related losses

├── eval/                          # Metric computation and result conversion

│   ├── evaluate.py                # Evaluate arbitrary prediction / GT / mask folders

│   ├── compare_results.py         # Dataset-specific comparison workflow

│   ├── constants.py               # Dataset definitions for evaluation

│   ├── topo_metric.py             # Topology metrics

│   └── resultsxls.py              # JSON-to-Excel conversion helper

├── detailed_evaluation/           # Extended topology and significance evaluation

├── crossing/                      # Vessel crossing / endpoint analysis utilities

├── micro_vessel/                  # Micro-vessel ROI analysis utilities

├── data_processing/               # Dataset conversion and distribution helpers

├── preprocessing.py               # Fundus enhancement and ROI-mask erosion

├── generate_mask.py               # Field-of-view mask generation

├── recolor_av3_labels.py          # Convert / recolor A/V labels

├── result.png                     # Example prediction figure

├── requirements.txt

├── LICENSE

└── README.md
```

---

## Open-Source Notes

Before redistributing this repository, please review the following:

- Remove private or non-redistributable data, including private large-field datasets and local experiment outputs.
- Respect licenses for RITE/DRIVE, HRF, LES-AV, GAVE, Fundus-AVSeg, and any other third-party datasets.
- Move large artifacts such as `train/__training/`, `eval/_Evaluation_Data/`, `eval/__results*/`, `.pth`, `.zip`, and `.xlsx` files to release assets or external storage.
- Replace hard-coded absolute paths in configuration files or scripts with relative paths or command-line arguments.
- Verify the license compatibility of copied baseline network implementations.
- If the local paper PDF is not redistributable, remove it and link to the DOI instead.

Suggested `.gitignore` entries:

```gitignore

train/_Data/

train/__training/

eval/_Evaluation_Data/

eval/__results*/

large_field_private/

*.zip

*.pth

*.xlsx

__pycache__/

*.pyc
```

---

## Citation

If you find this repository useful in your research, please cite our paper:

```bibtex

@article{zeng2026channelcoupling,

title = {Improve retinal artery/vein classification via channel coupling},

author = {Zeng, Shuang and Lee, Chee Hong and Li, Kaiwen and Xie, Boxu and Fu, Ourui and He, Hangzhou and Zhu, Lei and Lu, Yanye and Cheng, Fangxiao},

journal = {Expert Systems With Applications},

volume = {305},

pages = {130795},

year = {2026},

doi = {10.1016/j.eswa.2025.130795}

}
```

Please also cite the original RRWNet work if you use the recursive refinement architecture or its evaluation protocol.

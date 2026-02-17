# SAR and Optical Imagery for Dynamic Global Surface Water Monitoring

This repository contains the source code for the paper:

> **SAR and optical imagery for dynamic global surface water monitoring: addressing sensor-specific uncertainty for data fusion**  
> **Authors:** Davide Festa, Muhammed Hassaan, Wolfgang Wagner  
> *Manuscript under review*

![](figures/graph_abstract.jpg)

---

## Table of Contents
- [Repository Status](#repository-status)
- [Scientific Overview](#scientific-overview)
- [Dataset](#dataset)
  - [Description](#description)
  - [Data Preparation](#data-preparation)
  - [Normalization](#normalization)
- [Model Setup](#model-setup)
  - [Training Configuration](#training-configuration)
- [Pretrained Models](#pretrained-models)
- [Reproducibility and Scope](#reproducibility-and-scope)
- [Planned Improvements](#planned-improvements)
- [Citation](#citation)
- [Contact](#contact)

---

## Repository Status
**Work in progress.**  
The repository is under active development and will be updated alongside the review process.

---

## Scientific Overview

This framework addresses the limitations of single-sensor surface water monitoring by fusing independent uncertainty-aware deep learning models trained on multi-mission satellite data.

Key aspects:

- Independent U-Net models trained on **Sentinel-1 SAR** and **Sentinel-2 optical** imagery.
- **Bayesian deep learning** with Monte Carlo dropout for pixel-wise uncertainty.
- **Probabilistic data fusion** to mitigate sensor-specific failure modes.
- **Sensor-specific exclusion masks** to improve fusion robustness.
- Evaluation showing that **cloud-free optical-only assessments bias performance estimates**, emphasizing the need for multi-mission monitoring.

---

## Dataset

Experiments are based on the **S1S2-Water dataset**, a global reference dataset for surface water segmentation.

- Dataset: https://zenodo.org/records/11278238  
- Repository: https://github.com/MWieland/s1s2_water  
- Paper: https://ieeexplore.ieee.org/document/10321672  

### Description

The dataset contains **65 globally distributed samples**, each including:

- Sentinel-1 SAR (VV, VH)
- Sentinel-2 optical (Blue, Green, Red, NIR, SWIR1, SWIR2)
- Binary water masks
- Valid pixel masks
- DEM elevation and slope
- STAC-compliant metadata

Each sample represents a **100 × 100 km Sentinel-2 tile** stored as Cloud Optimized GeoTIFFs.

---

### Data Preparation

- All images were divided into **256 × 256 non-overlapping patches**.
- Patches were aligned across sensors.
- Tiles with invalid or missing data were excluded using valid-pixel masks.

---

### Normalization

Input data were standardized using **z-score normalization**.

Statistics were computed on the **training set only** and applied to validation and test sets.

Separate statistics were used for:

- Sentinel-1: VV, VH
- Sentinel-2: Blue, Green, Red, NIR

Normalization parameters are stored in:

      data/stats/
      ├── s1_mean.npy
      ├── s1_std.npy
      ├── s2_mean.npy
      └── s2_std.npy


These files are automatically loaded during training and inference.

---

## Model Setup

Two independent **U-Net models** were trained:

- **Sentinel-1 model:** VV, VH, slope
- **Sentinel-2 model:** Blue, Green, Red, NIR, slope

Both use a modified encoder–decoder architecture with skip connections and a final **1×1 convolution** producing a binary water mask.

Training included basic augmentations (brightness/contrast, scaling, flipping) and a **weighted BCE + Dice loss** to address class imbalance.

---

### Training Configuration

| Parameter        | Value               |
|------------------|---------------------|
| Architecture     | Modified U-Net      |
| Input patch size | 256 × 256           |
| Optimizer        | AdamW               |
| Learning rate    | 1e-3                |
| Weight decay     | 1e-2                |
| Batch size       | 64                  |
| Max epochs       | 100                 |
| Early stopping   | 15 epochs           |
| LR reduction     | ×0.1 after 5 epochs |
| Loss function    | Weighted BCE + Dice |
| Precision        | Mixed precision     |
| Framework        | PyTorch             |

---

## Pretrained Models

Pretrained checkpoints are not publicly released at this stage, as the manuscript is under review.

Researchers interested in reproducing the results or evaluating the models are encouraged to contact the authors directly. Access to the weights can be provided upon reasonable request.

---

## Reproducibility and Scope

This repository aims to support transparency and reproducibility of the proposed methodology.

All experiments described in the paper can be reproduced using the provided code and configurations, subject to minor stochastic variations and ongoing refactoring.

---

## Planned Improvements

- Improved documentation and installation instructions
- Example notebooks and end-to-end workflows
- Code reorganization and cleanup
- Release of pretrained model checkpoints

---

## Citation

If you use this repository, please cite:

```bibtex
@article{festa2026sar_optical_water,
  title   = {SAR and optical imagery for dynamic global surface water monitoring: addressing sensor-specific uncertainty for data fusion},
  author  = {Festa, Davide and Hassaan, Muhammed and Wagner, Wolfgang},
  journal = {Under review},
  year    = {2026}
}
```


## Contact

For questions about the code or methodology, please open an issue or contact the authors.

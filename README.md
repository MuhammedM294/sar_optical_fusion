# SAR and Optical Imagery for Dynamic Global Surface Water Monitoring  

This repository contains the source code for the paper:

> **SAR and optical imagery for dynamic global surface water monitoring: addressing sensor-specific uncertainty for data fusion**  
> **Authors:** Davide Festa, Muhammed Hassaan, Wolfgang Wagner  
> *Manuscript under review* 

![](figures/graph_abstract.jpg)
## Repository Status: Work in Progress
This repository is currently under active development.

---

## Scientific Overview

The proposed framework addresses the limitations of single-sensor surface water monitoring by fusing independent uncertainty-aware deep learning models trained on multi-mission satellite data.

Key methodological aspects include:

- Independent U-Net models trained on **Sentinel-1 SAR** and **Sentinel-2 optical** imagery for surface water mapping.
- **Bayesian deep learning** with Monte Carlo dropout to explicitly model pixel-wise predictive uncertainty.
- **Probabilistic data fusion** that mitigates sensor-specific failure modes and improves global surface water retrieval accuracy.
- **Sensor-specific exclusion masks** that enhance fusion robustness while explicitly conveying sensor-dependent limitations.
- An evaluation strategy demonstrating that **cloud-free optical-only assessments bias performance estimates**, highlighting the necessity of representative multi-mission operational monitoring.

---

## Dataset

The experiments in this study are based on the **S1S2-Water dataset**, a global reference dataset designed for training and evaluating deep learning models for surface water segmentation from multi-sensor satellite imagery.

- **Dataset download:** https://zenodo.org/records/11278238  
- **Official repository:** https://github.com/MWieland/s1s2_water
- **Paper:** https://ieeexplore.ieee.org/document/10321672

### Description

S1S2-Water consists of **65 globally distributed samples**, each containing:

- **Sentinel-1 SAR imagery** (VV, VH)
- **Sentinel-2 optical imagery** (Blue, Green, Red, NIR, SWIR1, SWIR2)
- **Quality-checked binary water masks**
- **Valid pixel masks**
- **Copernicus DEM elevation and slope**
- **STAC-compliant metadata**

The samples are selected across diverse climate zones, land-cover types, and hydrological conditions to support **robust global surface water modeling**.

Each sample corresponds to a **100 × 100 km Sentinel-2 tile**, stored as Cloud Optimized GeoTIFFs in a common UTM projection.

### Data Preparation

All Sentinel-1 and Sentinel-2 images were divided into **256 × 256 pixel patches** prior to model training and inference.  
Patches were extracted from the original scenes following the dataset tiling scheme, ensuring consistent spatial resolution and alignment between sensors.

Tiles containing invalid or missing data were excluded based on the provided valid-pixel masks.

### Normalization

Input data were normalized using **z-score normalization**:

   

The normalization statistics were computed over the **training set** separately for:

- Sentinel-1 bands (VV, VH)
- Sentinel-2 bands (Blue, Green, Red, NIR)

The resulting mean and standard deviation arrays are provided in:


      data/
      └── stats/
      ├── s1_mean.npy
      ├── s1_std.npy
      ├── s2_mean.npy
      └── s2_std.npy
These files are automatically loaded during training and inference.

## Pretrained Models

The pretrained model checkpoints used in the experiments are not publicly released at this stage, as the manuscript is currently under review.

Researchers interested in reproducing the results or evaluating the models are encouraged to contact the authors directly. Access to the pretrained weights can be provided upon reasonable request.

## Reproducibility and Scope

The repository is intended to support transparency, reproducibility, and reuse of the proposed methodology.
All experiments reported in the paper can be reproduced using the provided code and configurations, subject to minor variations due to stochastic training and ongoing refactoring.

---
## Planned Improvements

- Improved documentation and installation instructions  
- Example notebooks and end-to-end workflows  
- Code reorganization and cleanup  
- Release of pretrained model checkpoints  

---

## Citation

If you use this repository, please cite the paper:

```bibtex
@article{festa2026sar_optical_water,
  title   = {SAR and optical imagery for dynamic global surface water monitoring: addressing sensor-specific uncertainty for data fusion},
  author  = {Festa, Davide and Hassaan, Muhammed and Wagner, Wolfgang},
  journal = {Under review},
  year    = {2026}
}
```

## Contact

For questions regarding the code or methodology, please open an issue or contact the authors.









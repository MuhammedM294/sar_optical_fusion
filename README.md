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

The proposed framework addresses the limitations of single-sensor surface water monitoring by by fusing independent uncertainty-aware deep learning models trained on multi-mission satellite data.

Key methodological aspects include:

- Independent U-Net models trained on **Sentinel-1 SAR** and **Sentinel-2 optical** imagery for surface water mapping.
- **Bayesian deep learning** with Monte Carlo dropout to explicitly model pixel-wise predictive uncertainty.
- **Probabilistic data fusion** that mitigates sensor-specific failure modes and improves global surface water retrieval accuracy.
- **Sensor-specific exclusion masks** that enhance fusion robustness while explicitly conveying sensor-dependent limitations.
- An evaluation strategy demonstrating that **cloud-free optical-only assessments bias performance estimates**, highlighting the necessity of representative multi-mission operational monitoring.

---

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

## Contact

For questions regarding the code or methodology, please open an issue or contact the authors.



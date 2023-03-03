## Interpretable Deep Learning-based Solar Flare Prediction with post hoc Attention for Operational Forecasting

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE) 
[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![jupyter](https://img.shields.io/badge/Jupyter-Lab-F37626.svg?style=flat&logo=Jupyter)](https://jupyterlab.readthedocs.io/en/stable)
[![pytorch](https://img.shields.io/badge/PyTorch-1.6.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

This repo presents a post hoc analysis of a deep learning-based full-disk solar flare prediction model using three post hoc attention methods: (i) Guided Gradient-weighted Class Activation Mapping, (ii) Deep Shapley Additive Explanations, and (iii) Integrated Gradients.

### Source Code Documentation

#### 1. download_mag:
This folder/package contains one python module, "download_jp2.py". There are two functions inside this module. First Function: "download_from_helioviewer()" downloads jp2 magnetograms from helioveiwer api : Helioviewer Second Function: "jp2_to_jpg_conversion()" converts jp2s to jpgs for faster computation. If resize=True, pass height and width to resize the magnetograms



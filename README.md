## Explainable Deep Learning-based Solar Flare Prediction with post hoc Attention for Operational Forecasting

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE) 
[![python](https://img.shields.io/badge/Python-3.7.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?style=flat&logo=Jupyter)](https://jupyterlab.readthedocs.io/en/stable)
[![pytorch](https://img.shields.io/badge/PyTorch-1.10.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Docs - GitHub.io](https://img.shields.io/static/v1?logo=captum&style=flat&color=pink&label=lib&message=captum-0.5.0)](https://captum.ai/)

This repo presents a post hoc analysis of a deep learning-based full-disk solar flare prediction model using three post hoc attention methods: (i) Guided Gradient-weighted Class Activation Mapping, (ii) Deep Shapley Additive Explanations, and (iii) Integrated Gradients.
### Architecture of Our AlexNet CNN Pipeline based Full-disk Model

![alt text](https://github.com/chetrajpandey/explainingFullDisk/blob/main/readme_resoc/model.pdf?raw=true)

### Source Code Documentation

#### 1. download_mag:
This folder/package contains one python module, "download_jp2.py". There are two functions inside this module. First Function: "download_from_helioviewer()" downloads jp2 magnetograms from helioveiwer api : Helioviewer Second Function: "jp2_to_jpg_conversion()" converts jp2s to jpgs for faster computation. If resize=True, pass height and width to resize the magnetograms

#### 2. data_labeling:
Run python labeling.py : Contains functions to generate labels, binarize, filtering files, and creating 4-fold CV dataset.
Reads goes_integrated_flares.csv files from data_source.
Generated labels are stored inside data_labels. 
labeling.py generates labels with multiple columns that we can use for post result analysis. Information about flares locations, and any other flares that occured with in the period of 24 hours.
For simplification:  folder inside data_labels, named simplified_data_labels that contains two columns: the name of the file and actual target that is sufficient to train the model.


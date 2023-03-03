# Interpretable Deep Learning-based Solar Flare Prediction with post hoc Attention for Operational Forecasting
This repo presents a post hoc analysis of a deep learning-based full-disk solar flare prediction model using three post hoc attention methods: (i) Guided Gradient-weighted Class Activation Mapping, (ii) Deep Shapley Additive Explanations, and (iii) Integrated Gradients.

##Source Code Documentation

### 1. download_mag:
This folder/package contains one python module, "download_jp2.py". There are two functions inside this module. First Function: "download_from_helioviewer()" downloads jp2 magnetograms from helioveiwer api : Helioviewer Second Function: "jp2_to_jpg_conversion()" converts jp2s to jpgs for faster computation. If resize=True, pass height and width to resize the magnetograms



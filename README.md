## Explainable Deep Learning-based Solar Flare Prediction with post hoc Attention for Operational Forecasting

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE) 
[![python](https://img.shields.io/badge/Python-3.7.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?style=flat&logo=Jupyter)](https://jupyterlab.readthedocs.io/en/stable)
[![pytorch](https://img.shields.io/badge/PyTorch-1.10.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Docs - GitHub.io](https://img.shields.io/static/v1?logo=captum&style=flat&color=pink&label=lib&message=captum-0.5.0)](https://captum.ai/)

This repo presents a post hoc analysis of a deep learning-based full-disk solar flare prediction model using three post hoc attention methods: (i) Guided Gradient-weighted Class Activation Mapping, (ii) Deep Shapley Additive Explanations, and (iii) Integrated Gradients.
### Architecture of Our AlexNet CNN Pipeline based Full-disk Model

![alt text](https://github.com/chetrajpandey/explainingFullDisk/blob/main/readme_resoc/model.png?raw=true)

### Source Code Documentation

#### 1. download_mag:
This folder/package contains one python module, "download_jp2.py". There are two functions inside this module. First Function: "download_from_helioviewer()" downloads jp2 magnetograms from helioveiwer api : Helioviewer Second Function: "jp2_to_jpg_conversion()" converts jp2s to jpgs for faster computation. If resize=True, pass height and width to resize the magnetograms

#### 2. data_labeling:
Run python labeling.py : Contains functions to generate labels, binarize, filtering files, and creating 4-fold CV dataset.
Reads goes_integrated_flares.csv files from data_source.
Generated labels are stored inside data_labels. 
labeling.py generates labels with multiple columns that we can use for post result analysis. Information about flares locations, and any other flares that occured with in the period of 24 hours.
For simplification:  folder inside data_labels, named simplified_data_labels that contains two columns: the name of the file and actual target that is sufficient to train the model.

#### 3. modeling:

(a) model.py: This module contains the architecture of our model which can integrate the initial added convolutional layers to the pretrained AlexNet. Passing train=True utilizes the logsoftmax on the final activation. To get the probabilities during model predictions, pass train=False, and apply softmax to obtain the probabilities.<br /> 
(b) dataloader.py: This contains custom-defined data loaders for loading FL and NF class for selected augmentations.<br /> 
(c) evaluation.py: This includes functions to convert tensors to sklearn compatible array to compute confusion matrix. Furthermore TSS and HSS skill scores definition.<br /> 
(d) train.py: This module is the main module to train the model. Uses argument parsers for parameters change. This has seven paramters to change the model configuration:<br /> 
(i) --fold: choose from 1 to 4, to run the corresponding fold in 4CV-cross validation; default=1<br /> 
(ii) --epochs: number of epochs; default=40<br /> 
(iii) --batch_size: default=64<br /> 
(iv) --lr: initial learning rate selection; default=0.0099<br /> 
(v) --weight_decay: regularization parameter used by the loss function; default=0.01<br /> 
(vi) --patience: lr scheduler parameter used to reduce learning rate at specified value which indicates the fold tolerance; default=4<br /> 
(vii) --factor: lr scheduler parameter determining the quantitiy by which the learning rate is to be reduced; default=0.03<br /> 

For Example: <br /> 
To run the first fold for 50 epochs:<br /> 
python train.py --fold=1 --epochs=50 <br /> 
To run the second fold for 10 epochs:<br /> 
python train.py --fold=2 --epochs=10 <br/>

(e) predict_each_instance.py: This python module can be used to issue predictions for each instances using models trained in corresponding folds with their validation sets. Make sure to have train=False while loading the model to get probability scores.

#### 4. post_hoc:

This folder contains 1 jupyter notebook for Location Analysis indicating correctly/incorrectly made predictions in central and near-limb locations. <br /> 
Also for two hand-picked instances (one for True Positive, and one for False Positive), we show the generated post hoc attentions.



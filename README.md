# SS3D

Code for "SS3D: Unsupervised Out-of-Distribution Detection and Localization for Medical Volumes", submitted to the 2021 MICCAI Medical Out-of-Distribution Analysis Challenge. 

This repository contains the EfficientNet version with pretrained weights. For the actual challenge submission, weights obtained through contrastive learning were used instead, conform its restrictions.

Based on https://github.com/MIC-DKFZ/mood

For example, run the whole method on brain data with:
```
sudo python3 src/ss3d.py -r all -t <path to test data> -d <path to train data> -p <path to prediction folder> -s brain
```

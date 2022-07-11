# SS3D

Code for ["SS3D: Unsupervised Out-of-Distribution Detection and Localization for Medical Volumes"](https://link.springer.com/chapter/10.1007/978-3-030-97281-3_17), published in Biomedical Image Registration, Domain Generalisation and Out-of-Distribution Analysis of MICCAI 2021.

This repository contains the EfficientNet version with pretrained weights. For the actual challenge submission, weights obtained through contrastive learning were used instead, conform its restrictions.

Based on https://github.com/MIC-DKFZ/mood

For example, run the method on brain data with:
```
sudo python3 src/ss3d.py -r all -t <path to test data> -d <path to train data> -p <path to prediction folder> -s brain
```

### Citation

If you find this work helpful, consider citing it using

```
@inproceedings{doorenbos2021ss3d,
  title={SS3D: Unsupervised Out-of-Distribution Detection and Localization for Medical Volumes},
  author={Doorenbos, Lars and Sznitman, Raphael and M{\'a}rquez-Neila, Pablo},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={111--118},
  year={2021},
  organization={Springer}
}
```

# Perspective Phase Angle Model for Polarimetric 3D Reconstruction

The [paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620387.pdf) was accepted to ECCV 2022.

```
@INPROCEEDINGS{ChenPPAECCV2022,
author = {Guangcheng Chen and Li He and Yisheng Guan and Hong Zhang},
title = {Perspective Phase Angle Model for Polarimetric 3D Reconstruction},
booktitle = {European Conference on Computer Vision (ECCV)},
year = {2022}
}
```

## Introduction

This repository contains a demo of single-view planar normal estimation for ECCV 2022 paper -Perspective Phase Angle Model for Polarimetric 3D Reconstruction-.


<img src="figs/two_models.png" style="zoom: 20%;" />



The PPA model defines the polarization phase angle as the direction of the intersecting line of the image plane and the plane of incident (the red plane in the above figure), and hence allows the perspective effect to be considered in estimating surface normals from the phase angles and in defining the constraint on surface normal by the phase angle.

<img src="figs/demo_2.gif" style="zoom: 50%;" />



## Environments

```
pip install opencv-python numpy skimage
```


## Normal estimation

### single-view demo

The following script shows how to use the PPA model for estimating a planar surface normal.

```python
python singleview_normal.py
```

### normal from depth demo
The example code `normal_from_depth.py` shows how to generate normal map from depth map and generate phase angle map from normal map.

```
python normal_from_depth.py
```

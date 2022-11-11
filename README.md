# Perspective Phase Angle Model for Polarimetric 3D Reconstruction

The [paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620387.pdf) was accepted to ECCV 2022.

```
@InProceedings{10.1007/978-3-031-20086-1_23,
author="Chen, Guangcheng
and He, Li
and Guan, Yisheng
and Zhang, Hong",
editor="Avidan, Shai
and Brostow, Gabriel
and Ciss{\'e}, Moustapha
and Farinella, Giovanni Maria
and Hassner, Tal",
title="Perspective Phase Angle Model for Polarimetric 3D Reconstruction",
booktitle="Computer Vision -- ECCV 2022",
year="2022",
publisher="Springer Nature Switzerland",
address="Cham",
pages="398--414",
abstract="Current polarimetric 3D reconstruction methods, including those in the well-established shape from polarization literature, are all developed under the orthographic projection assumption. In the case of a large field of view, however, this assumption does not hold and may result in significant reconstruction errors in methods that make this assumption. To address this problem, we present the perspective phase angle (PPA) model that is applicable to perspective cameras. Compared with the orthographic model, the proposed PPA model accurately describes the relationship between polarization phase angle and surface normal under perspective projection. In addition, the PPA model makes it possible to estimate surface normals from only one single-view phase angle map and does not suffer from the so-called {\$}{\$}{\backslash}pi {\$}{\$}$\pi$-ambiguity problem. Experiments on real data show that the PPA model is more accurate for surface normal estimation with a perspective camera than the orthographic model.",
isbn="978-3-031-20086-1"
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

'''
File: singleview_normal.py
Project: ppa4p3d
Created Date: 2022-07-18
Author: Guangcheng Chen (GcC)
Email : 2112001004@mail2.gdut.edu.cn
Summary: demo of using the PPA model for single-view normal estimation
'''

import os
opj = os.path.join
import numpy as np
import cv2
from skimage.draw import polygon as skd_polygon
from utils_pol import calc_avg, calc_dop, calc_aop, quad_undistort, quad_blur, quad_resize, gamma_correct
from utils_3d import pi, rays_from_pixels
from misc import read_camera, generate_sample_list
import time
import glob


def normal_from_eigen(M):
    EW, EV = np.linalg.eig(M.T.dot(M))
    n = EV.T[np.argmin(EW)].reshape([3,1])
    n_est = n if n[2] <= 0 else -n
    return n_est

def normal_from_reflection_AoP(AoP_high_DoP, rays_high_DoP):
    """
    Args:
        AoP_high_DoP ([type]): Nx1
        rays_high_DoP ([type]): Nx1
    Returns:
        [type]: [description]
    """
    M = np.empty([len(rays_high_DoP), 3], dtype=np.float32)
    vx = rays_high_DoP[:,0]
    vy = rays_high_DoP[:,1]
    vz = rays_high_DoP[:,2]
    varphi = AoP_high_DoP + np.pi/2
    cos_varphi = np.cos(varphi)
    sin_varphi = np.sin(varphi)
    M[:, 0] = -vz * sin_varphi
    M[:, 1] = -vz * cos_varphi
    M[:, 2] = (vy * cos_varphi + vx * sin_varphi)
    n_est = normal_from_eigen(M)
    return n_est, rays_high_DoP, AoP_high_DoP, M

# Load and preprocess data
## configuration
idx_start = 0
num_views = -1
th_DoP = 0.1

## data paths
list_path_images = glob.glob("data/single_normal/images/**.*png")

## camera parameters
GAMMA = 0.5
list_path_cameras = glob.glob("data/single_normal/cams/**.*txt")
list_cameras = []
for path_cam_txt in list_path_cameras:
    list_cameras.append(read_camera(path_cam_txt))
CAMERA_MATRIX = list_cameras[0][1]
DIST_COEFFS = np.array([0,0,0,0,0.])

RESIZE_FACTOR = 1.
camera_matrix_resize = np.array(CAMERA_MATRIX)
camera_matrix_resize[:2] = camera_matrix_resize[:2]*RESIZE_FACTOR

## plane parameters
pts_3d_corners_w = np.load("data/single_normal/pts_3d_corners_w.npy")
normal_w = np.load("data/single_normal/normal_w.npy")

## main part
nd_normal_est = np.empty([0,3],dtype=np.float32)
nd_normal_gt = np.empty([0,3],dtype=np.float32)
for idx, path_image in enumerate(list_path_images):
    print("#{:0>3d}".format(idx))

    time_start = time.time()
    ### preprocess
    raw = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
    raw_blur = quad_blur(raw, ksize=(5,5))
    raw_resize = quad_resize(raw_blur, resize=(np.array(raw_blur.shape)/2.*RESIZE_FACTOR).astype(np.int32))
    raw_undist = quad_undistort(raw_resize, camera_matrix_resize, DIST_COEFFS)

    ### image
    polimage = gamma_correct(raw_undist, GAMMA)
    avgimage = calc_avg(raw_undist.astype(np.float32)).astype(np.uint8)
    h,w = avgimage.shape

    ### read poses
    extrinsics, intrinsics, _,_,_,_ = list_cameras[idx]
    Rcw = extrinsics[:, :3]
    tcw = extrinsics[:,3:4]
    ### ground-truth normal
    normal_c = Rcw.dot(normal_w)
    normal_c = normal_c if normal_c[2] < 0 else -normal_c
    gt_plane_normal_c = normal_c
    
    ### image mask
    mask_gt = np.zeros([h, w],dtype=bool)
    corners_3d = Rcw.dot(pts_3d_corners_w.T).T + tcw.T
    corners_3d_mean = corners_3d.mean(axis=0)
    corners_3d_centralized = corners_3d - corners_3d_mean
    corners_3d_shrink = corners_3d_centralized * 0.9 + corners_3d_mean
    corners_2d = pi(corners_3d_shrink, camera_matrix_resize)

    c_ = corners_2d[:,0]
    r_ = corners_2d[:,1]
    rr, cc = skd_polygon(r_, c_,  shape=mask_gt.shape)
    mask_gt[rr, cc] = True
    print(" preprocess time: {:.6f}".format(time.time() - time_start))

    ### planar normal estimation
    AoP, mask_aop_nan = calc_aop(polimage.astype(np.float32), return_mask=True)
    DoP = calc_dop(polimage.astype(np.float32))
    mask_dop = DoP > th_DoP
    mask_aop_dop = mask_dop
    mask_aop_dop = (mask_gt>0) & mask_aop_dop
    
    pts2D_masked = np.argwhere(mask_aop_dop)[::1, [1,0]]  # image coordinates
    rays_masked = rays_from_pixels(pts2D_masked, camera_matrix_resize)
    AoP_masked = AoP[mask_aop_dop].ravel()
    
    time_start = time.time()
    n_pred, rays_high_DoP, AoP_high_DoP, M = normal_from_reflection_AoP(AoP_masked, rays_masked)
    print("calculation time: {:.6f}".format(time.time() - time_start))

    nd_normal_est = np.append(nd_normal_est, n_pred.T, axis=0)
    nd_normal_gt  = np.append(nd_normal_gt, gt_plane_normal_c.T, axis=0)
    arccos_gt_est = np.arccos(gt_plane_normal_c.T.dot(n_pred))/np.pi*180
    print("   angular error: {:.6f} degree".format(arccos_gt_est[0,0]))

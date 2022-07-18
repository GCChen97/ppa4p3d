'''
File: utils_3d.py
Project: ppa4p3d
Created Date: 2022-07-18
Author: Guangcheng Chen (GcC)
Email : 2112001004@mail2.gdut.edu.cn
'''

import numpy as np
import cv2


def pi(points3D, camera_matrix):
    """projection function \pi

    Args:
        points3D (ndarray): Nx(x,y,z)
        camera_matrix (ndarray): camera matrix 3x3

    Returns:
        points2D_proj (ndarray): 2D image points
    """
    points2D_proj = points3D.dot(camera_matrix.T)
    points2D_proj = points2D_proj[:,:2] / points2D_proj[:,2:3]

    return points2D_proj

def rays_from_pixels(pixels_uv, camera_matrix, scale=1.):
    """
    Args:
        pixels_uv (ndarray): Nx(u,v)
        camera_matrix (nadarray): 3x3
        scale (float): used when scale image size
    Returns:
        rays (ndarray): Nx(x,y,z) normalized direction of each given pixel
    """
    if scale != 1.:
        camera_matrix_scaled = np.array(camera_matrix)
        camera_matrix_scaled[:2] *= scale
        camera_matrix = camera_matrix_scaled
    pts2D1 = cv2.convertPointsToHomogeneous(pixels_uv).squeeze(1).astype(np.float32)
    pts2D1 = pts2D1.dot(np.linalg.inv(camera_matrix).T)
    rays = pts2D1 / np.linalg.norm( pts2D1, axis=1, keepdims=True )
    return rays

def rays_of_all_pixels(image_shape, camera_matrix):
    h, w = image_shape
    U,V = np.meshgrid(np.arange(w), np.arange(h))
    uv_all = np.column_stack([U.ravel(),V.ravel()])
    rays_all = rays_from_pixels(uv_all, camera_matrix)
    return rays_all

def query_point_by_pixel(rays, plane_normal, plane_pos):
    """
    Notes:
        $
        l=\frac{\vec{n}^T(x_p-x_c)}{\vec{n}^T\vec{v}}\\
        x^*=x_c+l\vec{v}
        $
    Args:
        rays (ndarray): Nx[x,y,z] camera rays at camera frame
        plane_normal (ndarray): 3x1 normal at camera frame
        plane_pos (ndarray): 3x1 position at camera frame
    Returns:
        pts_cross (ndarray): Nx(x,y,z) positions of cross points between rays and given plane
    """
    neg_dist = plane_normal.T.dot(plane_pos)
    pts_cross = rays * neg_dist / ( rays.dot( plane_normal ) )

    return pts_cross

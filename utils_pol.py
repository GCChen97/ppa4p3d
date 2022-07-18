'''
File: utils_pol.py
Project: ppa4p3d
Created Date: 2022-07-18
Author: Guangcheng Chen (GcC)
Email : 2112001004@mail2.gdut.edu.cn
'''

import numpy as np
import cv2


def calc_avg(raw):
    """
    Args:
        raw: image data from polarization camera
    Returns:
        avg: average intensity
    """
    raw = raw.astype(np.float32)
    avg = (raw[::2,::2]+raw[::2,1::2]+raw[1::2,::2]+raw[1::2,1::2])/4
    return avg

def calc_aop(raw, return_mask=False):
    """
    Notes:
        note that input raw must be float data.
    Args:
        raw (ndarray): image data from polarization camera
        return_mask (bool): mask of unpolarized points
    Returns:
        AoP: angle of polarization/polarization phase angle
    """

    I0 = raw[1::2, 1::2]
    I45 = raw[::2, 1::2]
    I90 = raw[::2, ::2]
    I135 = raw[1::2, ::2]
    S1 = I0 - I90
    S2 = I45 - I135

    AoP = 0.5 * np.arctan2(S2, S1)
    AoP[AoP < -np.pi / 2] += np.pi
    AoP[AoP > np.pi / 2] -= np.pi
    if return_mask != True:
        return AoP
    else:
        nan_mask = (np.abs(S1)<1e-3) & (np.abs(S2)<1e-3)
        return AoP, nan_mask

def calc_dop(raw):
    """
    Args:
        raw: image data from polarization camera
    Returns:
        DoP: degree of polarization
    """

    I0 = raw[1::2, 1::2]
    I45 = raw[::2, 1::2]
    I90 = raw[::2, ::2]
    I135 = raw[1::2, ::2]
    S0 = (I0 + I90 )#+ I45 + I135)/2
    S1 = I0 - I90
    S2 = I45 - I135
    DoP = (S1 ** 2 + S2 ** 2) ** 0.5 / (S0+1e-6)
    return DoP

def quad_blur(raw, ksize=(3, 3), method='GaussianBlur'):
    """
    Args:
        raw: image data from polarization camera
        ksize: kernel size
        method: medianBlur/GaussianBlur
    Returns:
        ret: blured polarization image
    """
    if method=='GaussianBlur':
        q1 = cv2.GaussianBlur(raw[::2, ::2], ksize, 0)
        q2 = cv2.GaussianBlur(raw[::2, 1::2], ksize, 0)
        q3 = cv2.GaussianBlur(raw[1::2, ::2], ksize, 0)
        q4 = cv2.GaussianBlur(raw[1::2, 1::2], ksize, 0)
    elif method=='medianBlur':
        q1 = cv2.medianBlur(raw[::2, ::2], ksize[0], 0)
        q2 = cv2.medianBlur(raw[::2, 1::2], ksize[0], 0)
        q3 = cv2.medianBlur(raw[1::2, ::2], ksize[0], 0)
        q4 = cv2.medianBlur(raw[1::2, 1::2], ksize[0], 0)
    ret = np.zeros_like(raw)
    ret[::2, ::2] = q1
    ret[::2, 1::2] = q2
    ret[1::2, ::2] = q3
    ret[1::2, 1::2] = q4

    return ret

def quad_resize(raw, resize=[512, 612]):
    """
    Args:
        raw: image data from polarization camera
        resize: target size, (height, width)
    Returns:
        ret: resized polarization image
    """
    h0 = raw.shape[0] // 2
    w0 = raw.shape[1] //2
    hr = resize[0]
    wr = resize[1]
    if [h0, w0] == [hr, wr]:
        return raw
    else:
        ret = np.empty([resize[0]*2, resize[1]*2], dtype=np.uint8)

        q1 = cv2.resize(raw[::2, ::2],   (resize[1], resize[0]), cv2.INTER_CUBIC)
        q2 = cv2.resize(raw[::2, 1::2],  (resize[1], resize[0]), cv2.INTER_CUBIC)
        q3 = cv2.resize(raw[1::2, ::2],  (resize[1], resize[0]), cv2.INTER_CUBIC)
        q4 = cv2.resize(raw[1::2, 1::2], (resize[1], resize[0]), cv2.INTER_CUBIC)

        ret[::2, ::2] = q1
        ret[::2, 1::2] = q2
        ret[1::2, ::2] = q3
        ret[1::2, 1::2] = q4

        return ret

def quad_undistort(polimage, camera_matrix, dist_coeffs):
    polimage_undistort = np.empty_like(polimage)

    h, w = polimage.shape
    polimage_undistort[::2, ::2]   = cv2.undistort(polimage[::2,   ::2], camera_matrix, dist_coeffs)
    polimage_undistort[::2, 1::2]  = cv2.undistort(polimage[::2,  1::2], camera_matrix, dist_coeffs)
    polimage_undistort[1::2, ::2]  = cv2.undistort(polimage[1::2,  ::2], camera_matrix, dist_coeffs)
    polimage_undistort[1::2, 1::2] = cv2.undistort(polimage[1::2, 1::2], camera_matrix, dist_coeffs)
    
    return polimage_undistort

def gamma_correct(image, gamma):
    gamma_inv = 1 / gamma
    gamma_lut = np.arange(256)
    gamma_lut = (gamma_lut/255.)**(gamma_inv) * 255.

    return gamma_lut[image]


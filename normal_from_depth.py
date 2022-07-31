import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def readDepthDmb(path_dmb):
    with open(path_dmb,"rb") as dmb:
        type_ = int.from_bytes(dmb.read(4),byteorder="little",signed=True)

        h = int.from_bytes(dmb.read(4),byteorder="little",signed=True)
        w = int.from_bytes(dmb.read(4),byteorder="little",signed=True)
        nb = int.from_bytes(dmb.read(4),byteorder="little",signed=True)
        depth = np.frombuffer(dmb.read(),dtype=np.float32).reshape([h, w])
        dmb.close()
        return depth

def ReadCamera(path_cam_txt):
    str_cam = open(path_cam_txt, 'r')
    str_cam.readline()
    extrinsics = np.array([float(str_) for str_ in str_cam.readline()[:-2].split(" ")] + 
                          [float(str_) for str_ in str_cam.readline()[:-2].split(" ")] + 
                          [float(str_) for str_ in str_cam.readline()[:-2].split(" ")] ).reshape([3,4])
    str_cam.readline()
    str_cam.readline()
    str_cam.readline()
    intrinsics = np.array([float(str_) for str_ in str_cam.readline()[:-2].split(" ")] + 
                         [float(str_) for str_ in str_cam.readline()[:-2].split(" ")] + 
                         [float(str_) for str_ in str_cam.readline()[:-2].split(" ")] ).reshape([3,3])
    str_cam.readline()
    depth_min, interval, depth_num, depth_max = [float(str_) for str_ in str_cam.readline()[:-2].split(" ")]
    str_cam.close()
    return extrinsics, intrinsics, depth_min, interval, depth_num, depth_max

def ray_of_pixel(pixels_uv, camera_matrix, scale=1., normalized=True):
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
    pts2D1 = np.column_stack([pixels_uv, np.ones([len(pixels_uv), 1])])
    rays = pts2D1.dot(np.linalg.inv(camera_matrix).T)
    if normalized:
        rays = rays / np.linalg.norm( rays, axis=1, keepdims=True )
    return rays

def ray_of_image(image_shape, camera_matrix, normalized=True):
    h, w = image_shape
    U,V = np.meshgrid(np.arange(w), np.arange(h))
    uv_all = np.column_stack([U.ravel(),V.ravel()])
    rays_all = ray_of_pixel(uv_all, camera_matrix, normalized=True)
    return rays_all

def map_aop(raw):
    """
    Notes:
        please input raw in float
    Args:
        raw (ndarray): raw image from camera
    Returns:
        aop: angle of polarization
    """
    if not raw.dtype == np.float32 and not raw.dtype == np.float64:
        raise Exception("Invalid ndarray type:", raw.dtype)

    I0 = raw[1::2, 1::2]
    I45 = raw[::2, 1::2]
    I90 = raw[::2, ::2]
    I135 = raw[1::2, ::2]
    S1 = I0 - I90
    S2 = I45 - I135

    aop = 0.5 * np.arctan2(S2, S1)
    aop[aop < -np.pi / 2] += np.pi
    aop[aop > np.pi / 2] -= np.pi

    return aop

def gamma_correct(raw, gamma):
    if not raw.dtype == np.uint8:
        raise Exception("Invalid ndarray type:", raw.dtype)
    gamma_inv = 1 / gamma
    gamma_lut = np.arange(256)
    gamma_lut = (gamma_lut/255.)**(gamma_inv) * 255.
    if len(raw.shape) == 2:
        return gamma_lut[raw]
    elif len(raw.shape) == 3:
        ret = np.empty(raw.shape, dtype=np.float32)
        ret[..., 0] = gamma_lut[raw[..., 0]]
        ret[..., 1] = gamma_lut[raw[..., 1]]
        ret[..., 2] = gamma_lut[raw[..., 2]]
        return ret


path_dataset = os.path.dirname(__file__)
id_img = 51

image = cv2.imread(path_dataset+"/data/normal_from_depth/{:0>8}_raw.png".format(id_img), cv2.IMREAD_GRAYSCALE)
image = gamma_correct(image, 0.5)

depths = readDepthDmb(path_dataset+"/data/normal_from_depth/depth_p.dmb".format(id_img))

camera = ReadCamera(path_dataset+"/data/normal_from_depth/{:0>8}_cam.txt".format(id_img))
K = camera[1]
fx = K[0,0]
fy = K[1,1]
cx = K[0,2]
cy = K[1,2]

mask = cv2.imread(path_dataset+"/data/normal_from_depth/{:0>8}_mask.png".format(id_img), cv2.IMREAD_GRAYSCALE).astype(bool)

# Angle of Polarization
aop = map_aop(image)

# partial derivatives
dzdx = depths[:-1,1:] - depths[:-1,:-1]
dzdy = depths[1:,:-1] - depths[:-1,:-1]

# camera rays
rays = ray_of_image([1023, 1223], K).reshape([1023, 1223, 3])
vx = rays[..., 0]
vy = rays[..., 1]
vz = rays[..., 2]


# generate normal map from depth map
nx = -fy*dzdx
ny = -fx*dzdy
nz = (np.arange(1223).reshape([1,-1])-cx) * dzdx + (np.arange(1023).reshape([-1,1])-cy) * dzdy + depths[:-1,:-1]

N = np.zeros([1023,1223,3])
N[..., 0] = nx
N[..., 1] = ny
N[..., 2] = nz
## normalize the normal vector
N = N/np.linalg.norm(N, axis=2).reshape([1023,1223,1])
nx = N[..., 0]
ny = N[..., 1]
nz = N[..., 2]


# generate AoP from normal map and camera rays
phi = -np.arctan2(-vz*ny+vy*nz, -vz*nx+vx*nz)+np.pi/2
phi[phi<-np.pi/2] += np.pi
phi[phi> np.pi/2] -= np.pi
plt.imshow(phi,cmap="hsv")
plt.show()


# residual of ppa constraint
sinphi = np.sin(aop[:-1,:-1]+np.pi/2)
cosphi = np.cos(aop[:-1,:-1]+np.pi/2)

vz_sinphi = vz * sinphi
vz_cosphi = vz * cosphi
neg_vy_cosphi_plus_vx_sinphi = -(vy*cosphi+vx*sinphi)

residual = vz_sinphi * nx + vz_cosphi * ny + neg_vy_cosphi_plus_vx_sinphi * nz
residual[mask[:-1,:-1]==False] = np.inf
plt.imshow(np.abs(residual))
plt.show()


# residual given the ground-truth normal
gtx=-0.14363778
gty=-0.79288903
gtz=-0.59219521
residual = vz_sinphi * gtx + vz_cosphi * gty + neg_vy_cosphi_plus_vx_sinphi * gtz
residual[mask[:-1,:-1]==False] = np.inf
# plt.imshow(np.abs(residual))
# plt.show()

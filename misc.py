'''
File: misc.py
Project: ppa4p3d
Created Date: 2022-07-18
Author: Guangcheng Chen (GcC)
Email : 2112001004@mail2.gdut.edu.cn
'''

import numpy as np
from os.path import join as opj


def read_camera(path_cam_txt):
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

def write_camera():
    pass

def generate_sample_list(path_dense_folder):
    path_cluster_list = opj(path_dense_folder, "pair.txt")
    str_pair = open(path_cluster_list, 'r')
    num_images = int(str_pair.readline()[:-1])
    list_problems = []
    for i in range(num_images):
        problem = {"ref_image_id":-1, "src_image_ids":[]}
        ref_image_id = int(str_pair.readline()[:-1])
        problem["ref_image_id"] = ref_image_id
        tmp = [ int(str_) for str_ in str_pair.readline()[:-2].split(" ") ]
        tmp_ids = [ int_ for int_ in tmp[1::2] ]
        tmp_scores = [int_ for int_ in tmp[2::2] ]

        num_src_images = tmp[0]
        for i in range(num_src_images):
            if tmp_scores[i]>0:
                problem["src_image_ids"].append(tmp_ids[i])
        list_problems.append(problem)
    return list_problems

def readDepthDmb(path_dmb):
    with open(path_dmb,"rb") as dmb:
        type_ = int.from_bytes(dmb.read(4),byteorder="little",signed=True)

        h = int.from_bytes(dmb.read(4),byteorder="little",signed=True)
        w = int.from_bytes(dmb.read(4),byteorder="little",signed=True)
        nb = int.from_bytes(dmb.read(4),byteorder="little",signed=True)
        depth = np.frombuffer(dmb.read(),dtype=np.float32).reshape([h, w])
        dmb.close()
        return depth

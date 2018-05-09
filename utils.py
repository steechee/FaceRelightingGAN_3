from __future__ import print_function

import os
import math
import json
import logging
import numpy as np
from PIL import Image
from datetime import datetime

import tensorflow as tf

def prepare_dirs_and_logger(config):
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    if config.load_path:
        if config.load_path.startswith(config.log_dir):
            config.model_dir = config.load_path
        else:
            if config.load_path.startswith(config.dataset):
                config.model_name = config.load_path
            else:
                config.model_name = "{}_{}".format(config.dataset, config.load_path)
    else:
        config.model_name = "{}_{}".format(config.dataset, get_time())

    if not hasattr(config, 'model_dir'):
        config.model_dir = os.path.join(config.log_dir, config.model_name)
    config.data_path = os.path.join(config.data_dir, config.dataset)

    for path in [config.log_dir, config.data_dir, config.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

def save_config(config):
    param_path = os.path.join(config.model_dir, "params.json")

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def rank(array):
    return len(array.shape)

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, scale_each=False):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h:h+h_width, w:w+w_width] = tensor[k]
            k = k + 1
    return grid

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, scale_each=False):
    ndarr = make_grid(tensor, nrow=nrow, padding=padding,
                            normalize=normalize, scale_each=scale_each)
    im = Image.fromarray(ndarr)
    im.save(filename)


def templight(self):
    light = np.array([[-7.92065536e-02,  3.49902089e-02,  1.42030788e-01],
    [ 2.97175636e-02, -1.10210985e-03, -1.12216669e-02],
    [ 1.14490456e+00,  7.25263925e-01,  4.18665994e-01],
    [ 6.91308048e-02,  4.45049699e-02,  3.14903428e-02],
    [-9.65037690e-03, -3.04341827e-03, -9.75049582e-04],
    [ 1.76383977e-01,  2.28311868e-01,  2.23022992e-01],
    [-7.38700709e-01, -4.50987158e-01, -1.76474364e-01],
    [-8.12088808e-02, -4.98664871e-02, -3.19874631e-02],
    [-1.07426664e-01, -1.50386091e-01, -1.36360101e-01]])

    return light


def getmatrix(L):

    c1 = 0.429043
    c2 = 0.511664
    c3 = 0.743152
    c4 = 0.886227
    c5 = 0.247708

    # print L.get_shape()
    # M = [ [c1*L9, c1*L5, c1*L8, c2*L4],
    # [c1*L5   -c1*L9   c1*L6   c2*L2
    # c1*L8   c1*L6    c3*L7   c2*L3
    # c2*L4   c2*L2    c2*L3   c4*L1 - c5*L7 ]

    e0 = tf.constant([0,0,0,0,0,0,0,0,0], dtype = tf.float32)
    e1 = tf.constant([1,0,0,0,0,0,0,0,0], dtype = tf.float32)
    e2 = tf.constant([0,1,0,0,0,0,0,0,0], dtype = tf.float32)
    e3 = tf.constant([0,0,1,0,0,0,0,0,0], dtype = tf.float32)
    e4 = tf.constant([0,0,0,1,0,0,0,0,0], dtype = tf.float32)
    e5 = tf.constant([0,0,0,0,1,0,0,0,0], dtype = tf.float32)
    e6 = tf.constant([0,0,0,0,0,1,0,0,0], dtype = tf.float32)
    e7 = tf.constant([0,0,0,0,0,0,1,0,0], dtype = tf.float32)
    e8 = tf.constant([0,0,0,0,0,0,0,1,0], dtype = tf.float32)
    e9 = tf.constant([0,0,0,0,0,0,0,0,1], dtype = tf.float32)


    L = tf.cast(tf.diag(L),dtype=tf.float32)

    M11 = c1 * tf.matmul(tf.matmul([e9,e0,e0,e0],L),tf.transpose([e9,e0,e0,e0]))
    M12 = c1 * tf.matmul(tf.matmul([e5,e0,e0,e0],L),tf.transpose([e0,e5,e0,e0]))
    M13 = c1 * tf.matmul(tf.matmul([e8,e0,e0,e0],L),tf.transpose([e0,e0,e8,e0]))
    M14 = c2 * tf.matmul(tf.matmul([e4,e0,e0,e0],L),tf.transpose([e0,e0,e0,e4]))
    M21 = c1 * tf.matmul(tf.matmul([e0,e5,e0,e0],L),tf.transpose([e5,e0,e0,e0]))
    M22 = -c1 * tf.matmul(tf.matmul([e0,e9,e0,e0],L),tf.transpose([e0,e9,e0,e0]))
    M23 = c1 * tf.matmul(tf.matmul([e0,e6,e0,e0],L),tf.transpose([e0,e0,e6,e0]))
    M24 = c2 * tf.matmul(tf.matmul([e0,e2,e0,e0],L),tf.transpose([e0,e0,e0,e2]))
    M31 = c1 * tf.matmul(tf.matmul([e0,e0,e8,e0],L),tf.transpose([e8,e0,e0,e0]))
    M32 = c1 * tf.matmul(tf.matmul([e0,e0,e6,e0],L),tf.transpose([e0,e6,e0,e0]))
    M33 = c3 * tf.matmul(tf.matmul([e0,e0,e7,e0],L),tf.transpose([e0,e0,e7,e0]))
    M34 = c2 * tf.matmul(tf.matmul([e0,e0,e3,e0],L),tf.transpose([e0,e0,e0,e3]))
    M41 = c2 * tf.matmul(tf.matmul([e0,e0,e0,e4],L),tf.transpose([e4,e0,e0,e0]))
    M42 = c2 * tf.matmul(tf.matmul([e0,e0,e0,e2],L),tf.transpose([e0,e2,e0,e0]))
    M43 = c2 * tf.matmul(tf.matmul([e0,e0,e0,e3],L),tf.transpose([e0,e0,e3,e0]))
    M44 = c4 * tf.matmul(tf.matmul([e0,e0,e0,e1],L),tf.transpose([e0,e0,e0,e1])) - c5 * tf.matmul(tf.matmul([e0,e0,e0,e7],L),tf.transpose([e0,e0,e0,e7]))

    M = M11 + M12 + M13 + M14 + M21 + M22 + M23 + M24 + M31 + M32 + M33 + M34 + M41 + M42 + M43 + M44
    M = tf.cast(M,dtype=tf.float32)

    return M


def getshading(normal, light):
    # normal 100*64*64*3
    # light 100*10*3

    # print (normal.shape) 16 3 64 64
    # print (light.shape) 9 3

    normal = tf.transpose(normal,[0, 2, 3, 1])

    nSample = normal.shape[0] # batch_size 100
    nPixel = normal.shape[1]*normal.shape[2] # 64*64 = 4096

    Lr = light[:,0] # 100*10
    Lg = light[:,1]
    Lb = light[:,2]

    # Lr_bw = light[:,:,0] # 100*10
    # Lr = Lr_bw[:,0:9] # 100*9
    # Lg_bw = light[:,:,1]
    # Lg = Lg_bw[:,0:9]
    # Lb_bw = light[:,:,2]
    # Lb = Lb_bw[:,0:9]


    Ns = tf.reshape(normal,[nSample, nPixel, 3]) # 100*4096*3
    N_ext = tf.ones([nSample, nPixel, 1], dtype=tf.float32) # 100*4096*1
    Ns = tf.concat([Ns, N_ext], axis=-1) # 100*4096*4

    for idx in range(nSample):
        nt = Ns[idx] # 4096*4

        mr = getmatrix(Lr)
        mg = getmatrix(Lg)
        mb = getmatrix(Lb)

        sr = tf.matmul(nt,mr)*nt # 4096*4
        sg = tf.matmul(nt,mg)*nt
        sb = tf.matmul(nt,mb)*nt

        s1 = tf.reshape(tf.reduce_sum(sr,axis=-1), [1,64,64]) # should be > 0 but lr_bw[idx,9] constant often < 0
        s2 = tf.reshape(tf.reduce_sum(sg,axis=-1), [1,64,64])
        s3 = tf.reshape(tf.reduce_sum(sb,axis=-1), [1,64,64])

        s = tf.stack([s1,s2,s3],axis=3)

        if idx == 0:
            shading = s
        else:
            shading = tf.concat([shading,s],axis=0)

    return shading


# def strtoarray(queue, batch_size):
#     # input : 16 1
#     # output : 16 27 or 16 9 3 or 16 3 9
#
#     for idx in range(batch_size): # 16
#         splitcomma = queue[idx][0].split(',')
#         for i in range(len(splitcomma)): #27
#             if i == 0:
#                 element = float(splitcomma[i].split('[')[-1])
#                 array = np.array([element])
#             elif i > 0 and i != len(splitcomma)-1:
#                 element = np.array([float(splitcomma[i])])
#                 array = np.concatenate((array,element),axis=-1)
#             else:
#                 element = np.array([float(splitcomma[i].split(']')[0])])
#                 array = np.concatenate((array,element),axis=-1)
#         # print (array.shape) # (27,)
#         array = np.reshape(array,[len(splitcomma),1])
#         # print (array.shape) # (27,1)
#
#         if idx == 0:
#             out = array
#         else:
#             out = np.concatenate([out,array],axis=1)
#
#     # print (out.shape) # 27,16
#     out = np.transpose(out,[1,0])
#     # print (out.shape) # 16 27
#
#     return out

import os
from PIL import Image
from glob import glob
import tensorflow as tf
import numpy as np
from utils import strtoarray

def get_light_loader(root, batch_size, scale_size, data_format, split=None, is_grayscale=False, seed=None):
    dataset_name = os.path.basename(root) # data/celeba
    if dataset_name in ['CelebA'] and split:
        root = os.path.join(root, 'splits', split) # default is train, therefore data/celeba/train

    for ext in ["csv"]:
        lightpaths = sorted(glob("{}/light/*.{}".format(root, ext))) # csv

        if len(lightpaths) != 0:
            break


    Lfilename_queue = tf.train.string_input_producer(list(lightpaths), shuffle=False, seed=seed)
    Lreader = tf.TextLineReader()
    Lfilename, Ldata = Lreader.read(Lfilename_queue)
    light = tf.decode_csv(Ldata, record_defaults = [["[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]"]])
    # light = tf.decode_csv(Ldata, record_defaults = [["[0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n]"]])

    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size # 5000+3*16?

    Lqueue = tf.train.batch(
        [light], batch_size=batch_size,
        num_threads=1, capacity=capacity,
        name='synthetic_lights')

    # print Lqueue.get_shape() # 16 1?
    # print Lqueue
    # return tf.to_float(Lqueue)

    # out = strtoarray(Lqueue, batch_size)

    return Lqueue

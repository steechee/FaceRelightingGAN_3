import os
from PIL import Image
from glob import glob
import tensorflow as tf
import numpy as np

def get_loader(root, batch_size, scale_size, data_format, split=None, is_grayscale=False, seed=None):
    dataset_name = os.path.basename(root) # data/celeba
    if dataset_name in ['CelebA'] and split:
        root = os.path.join(root, 'splits', split) # default is train, therefore data/celeba/splits/train

    for ext in ["jpg", "png"]:
        rgbpaths = sorted(glob("{}/rgb/*.{}".format(root, ext))) # png
        normalpaths = sorted(glob("{}/normal/*.{}".format(root, ext))) # png
        maskpaths = sorted(glob("{}/mask/*.{}".format(root, ext))) # png

        if ext == "jpg":
            tf_decode = tf.image.decode_jpeg
        elif ext == "png":
            tf_decode = tf.image.decode_png

        if len(rgbpaths) != 0:
            break


    # for ext in ["csv"]:
    #     lightpaths = sorted(glob("{}/light/*.{}".format(root, ext))) # csv
    #
    #     if len(lightpaths) != 0:
    #         break

    with Image.open(rgbpaths[0]) as img:
        w, h = img.size
        shape = [h, w, 3]
        # lightshape = [9, 3]

    filename_queue = tf.train.string_input_producer(list(rgbpaths), shuffle=False, seed=seed)
    reader = tf.WholeFileReader()
    filename, data = reader.read(filename_queue)
    image = tf_decode(data, channels=3)


    Nfilename_queue = tf.train.string_input_producer(list(normalpaths), shuffle=False, seed=seed)
    Nfilename, Ndata = reader.read(Nfilename_queue)
    normal = tf_decode(Ndata, channels=3)

    Mfilename_queue = tf.train.string_input_producer(list(maskpaths), shuffle=False, seed=seed)
    Mfilename, Mdata = reader.read(Mfilename_queue)
    mask = tf_decode(Mdata, channels=3)

    # Lfilename_queue = tf.train.string_input_producer(list(lightpaths), shuffle=False, seed=seed)
    # Lreader = tf.TextLineReader()
    # Lfilename, Ldata = Lreader.read(Lfilename_queue)
    # light = tf.decode_csv(Ldata, record_defaults = [
    # [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
    # [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
    # [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]])


    if is_grayscale:
        pass
        # image = tf.image.rgb_to_grayscale(image)

    image.set_shape(shape)
    normal.set_shape(shape)
    mask.set_shape(shape)

    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size # 5000+3*16?

    rgbqueue, normalqueue, maskqueue = tf.train.batch(
    # rgbqueue, normalqueue, maskqueue, lightqueue = tf.train.batch(
        [image, normal, mask], batch_size=batch_size,
        # [image, normal, mask, light], batch_size=batch_size,
        num_threads=1, capacity=capacity, name='synthetic_inputs')
        # 3 16 3 64 64
    # print rgbqueue.get_shape() # 16 64 64 3
    # print normalqueue.get_shape()
    # print maskqueue.get_shape()


    if data_format == 'NCHW':
        # queue = tf.transpose(queue, [0, 3, 1, 2])
        rgbqueue = tf.transpose(rgbqueue, [0, 3, 1, 2])
        normalqueue = tf.transpose(normalqueue, [0, 3, 1, 2])
        maskqueue = tf.transpose(maskqueue, [0, 3, 1, 2])

    elif data_format == 'NHWC':
        pass
    else:
        raise Exception("[!] Unkown data_format: {}".format(data_format))

    # return tf.to_float(rgbqueue), tf.to_float(normalqueue), tf.to_float(maskqueue), tf.to_float(lightqueue)
    return tf.to_float(rgbqueue), tf.to_float(normalqueue), tf.to_float(maskqueue)

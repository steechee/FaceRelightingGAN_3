import os
from PIL import Image
from glob import glob
import tensorflow as tf
import numpy as np

def get_loader(root, batch_size, scale_size, data_format, split=None, is_grayscale=False, seed=None):
    dataset_name = os.path.basename(root) # data/celeba
    if dataset_name in ['CelebA'] and split:
        root = os.path.join(root, 'splits', split) # default is train, therefore data/celeba/train

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

    for ext in ["npy"]:
        lightpaths = sorted(glob("{}/light/*.{}".format(root, ext))) # png


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
    # print (Mfilename_queue.size)
    Mfilename, Mdata = reader.read(Mfilename_queue)
    # print (Mdata.get_shape())
    mask = tf_decode(Mdata, channels=3)

    # # print (len(lightpaths))
    # Lfilename_queue = tf.train.string_input_producer(list(lightpaths), shuffle=False, seed=seed)
    # # reader.read(Lfilename_queue)
    # Lfilename, Ldata = reader.read(Lfilename_queue)
    # # Ldata = np.load(Lfilename_queue)
    # # print (Ldata.get_shape())
    # light = tf.cast(Ldata,dtype=tf.uint8)
    # # light = np.load(list(lightpaths))

    # print lightpaths
    # print len(lightpaths) 40000

    if is_grayscale:
        pass
        # image = tf.image.rgb_to_grayscale(image)

    image.set_shape(shape)
    normal.set_shape(shape)
    mask.set_shape(shape)

    # print mask.shape # 64 64 3
    # print light.get_shape()
    # light.set_shape([9, 3])

    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size # 5000+3*16?

    queue = tf.train.shuffle_batch(
        # [image, normal, mask, light], batch_size=batch_size,
        [image, normal, mask], batch_size=batch_size,
        num_threads=4, capacity=capacity,
        min_after_dequeue=min_after_dequeue, name='synthetic_inputs')
        # 3 16 3 64 64

    # queue2 = tf.train.shuffle_batch(
    #     # [image, normal, mask, light], batch_size=batch_size,
    #     [light], batch_size=batch_size,
    #     num_threads=4, capacity=capacity,
    #     min_after_dequeue=min_after_dequeue, name='synthetic_lights')
        # 1 16 3 9
    # print
    # print queue2.shape # 16
    # queue = tf.train.shuffle_batch(
    #     [image], batch_size=batch_size,
    #     num_threads=4, capacity=capacity,
    #     min_after_dequeue=min_after_dequeue, name='synthetic_inputs')

    # if dataset_name in ['CelebA']:
    #     # queue = tf.image.crop_to_bounding_box(queue, 50, 25, 128, 128)
    #     # queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])
    # else:
    #     queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])
    #
    if data_format == 'NCHW':
        # queue = tf.transpose(queue, [0, 3, 1, 2])
        queue = tf.transpose(queue, [0, 1, 4, 2, 3])
        # print (queue2.get_shape())
        # queue2 = tf.transpose(queue2, [0, 1, 3, 2])

    elif data_format == 'NHWC':
        pass
    else:
        raise Exception("[!] Unkown data_format: {}".format(data_format))

    # print queue
    # return [tf.to_float(queue), tf.to_float(normal), tf.to_float(mask), tf.to_float(light)]
    # return tf.to_float(queue), tf.to_float(queue2)
    return tf.to_float(queue)

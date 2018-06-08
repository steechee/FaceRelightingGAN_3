import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from utils import getmatrix
from utils import getshading10
from utils import getpointshading
from utils import getpointrecon
from hyper_parameters import *

def GeneratorCNN(x, output_num, z_num, repeat_num, hidden_num, data_format, reuse):
    with tf.variable_scope("G", reuse=reuse) as vs:
        # print x.get_shape() # 16 3 64 64

        ## Encoder
        x = slim.conv2d(x, 64, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        # print x.get_shape() # 16 64 64 64
        x = slim.conv2d(x, 128, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        # print x.get_shape() # 16 128 64 64
        z = x = slim.conv2d(x, 128, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
        # print x.get_shape() # 16 128 32 32

        ## mask
        ## more encode
        # x = slim.conv2d(x, 128, 3, 1, activation_fn=tf.nn.elu, data_format=data_format) # 16 128 32 32
        # x = slim.conv2d(x, 128, 3, 2, activation_fn=tf.nn.elu, data_format=data_format) # 16 128 16 16

        z_m = z
        z_m = tf.reshape(z_m, [-1, np.prod([32, 32, 128])]) # 16 32768
        z_m = slim.fully_connected(z_m, 64, activation_fn=None) # 16 64

        ## decode mask
        num_output = int(np.prod([32, 32, 128]))
        x_m = slim.fully_connected(z_m, num_output, activation_fn=None) # 16 32768
        x_m = reshape(x_m, 32, 32, 128, data_format) # 16 128 16 16

        # x_m = slim.conv2d(x_m, 128, 3, 1, activation_fn=tf.nn.elu, data_format=data_format) # 16 128 16 16
        x_m = slim.conv2d(x_m, 128, 3, 1, activation_fn=tf.nn.elu, data_format=data_format) # 16 128 16 16
        # x_m = upscale(x_m, 2, data_format) # 16 128 32 32
        x_m = slim.conv2d(x_m, 128, 3, 1, activation_fn=tf.nn.elu, data_format=data_format) # 16 128 32 32
        x_m = upscale(x_m, 2, data_format) # 16 128 64 64
        x_m = slim.conv2d(x_m, 128, 3, 1, activation_fn=tf.nn.elu, data_format=data_format) # 16 128 64 64
        x_m = slim.conv2d(x_m, 1, 3, 1, activation_fn=tf.nn.elu, data_format=data_format) # 16 3 64 64
        maskout = tf.concat([x_m, x_m, x_m], 1)

        # ## more encode
        # x = slim.conv2d(x, 128, 3, 1, activation_fn=tf.nn.elu, data_format=data_format) # 16 128 32 32
        # x = slim.conv2d(x, 128, 3, 2, activation_fn=tf.nn.elu, data_format=data_format) # 16 128 16 16
        #
        # z_m = x
        # z_m = tf.reshape(z_m, [-1, np.prod([16, 16, 128])]) # 16 32768
        # z_m = slim.fully_connected(z_m, 64, activation_fn=None) # 16 64
        #
        # ## decode mask
        # num_output = int(np.prod([16,16,128]))
        # x_m = slim.fully_connected(z_m, num_output, activation_fn=None) # 16 32768
        # x_m = reshape(x_m, 16, 16, 128, data_format) # 16 128 16 16
        #
        # x_m = slim.conv2d(x_m, 128, 3, 1, activation_fn=tf.nn.elu, data_format=data_format) # 16 128 16 16
        # x_m = slim.conv2d(x_m, 128, 3, 1, activation_fn=tf.nn.elu, data_format=data_format) # 16 128 16 16
        # x_m = upscale(x_m, 2, data_format) # 16 128 32 32
        # x_m = slim.conv2d(x_m, 128, 3, 1, activation_fn=tf.nn.elu, data_format=data_format) # 16 128 32 32
        # x_m = upscale(x_m, 2, data_format) # 16 128 64 64
        # x_m = slim.conv2d(x_m, 128, 3, 1, activation_fn=tf.nn.elu, data_format=data_format) # 16 128 64 64
        # maskout = slim.conv2d(x_m, 3, 3, 1, activation_fn=tf.nn.elu, data_format=data_format) # 16 3 64 64



        ## normal residual_block
        z_n = z
        z_n = tf.transpose(z_n, [0, 2, 3, 1])
        z_n = normalresnet(z_n, reuse=False)
        # z_n = tf.transpose(z_n, [0, 3, 1, 2])

        ## albedo residual_block
        z_a = z
        z_a = tf.transpose(z_a, [0, 2, 3, 1])
        z_a = albedoresnet(z_a, 1, reuse=False)
        # z_a = tf.transpose(z_a, [0, 3, 1, 2])

        ## normal decoder
        # print x.get_shape() # 16 128 32 32
        # x_n = tf.transpose(z_n, [0, 2, 3, 1])
        # print x.get_shape() # 16 32 32 128
        # x_n = tf.image.resize_images(x_n, [64, 64]) # bilinear upsampling
        x_n = tf.image.resize_images(z_n, [64, 64]) # bilinear upsampling
        # print x.get_shape() # 16 64 64 128
        x_n = tf.transpose(x_n, [0, 3, 1, 2])
        # print x.get_shape() # 16 128 64 64
        x_n = slim.conv2d(x_n, 128, 1, 1, activation_fn=tf.nn.elu, data_format=data_format)
        # print x.get_shape() # 16 128 64 64
        x_n = slim.conv2d(x_n, 128, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        # print x.get_shape() # 16 128 64 64
        normalout = slim.conv2d(x_n, 3, 1, 1, activation_fn=tf.nn.elu, data_format=data_format)
        # print x.get_shape() # 16 3 64 64


        ## albedo decoder
        # x_a = tf.transpose(z_a, [0, 2, 3, 1])
        x_a = tf.image.resize_images(z_a, [64, 64]) # bilinear upsampling
        x_a = tf.transpose(x_a, [0, 3, 1, 2])
        x_a = slim.conv2d(x_a, 128, 1, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x_a = slim.conv2d(x_a, 128, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        albedoout = slim.conv2d(x_a, 3, 1, 1, activation_fn=tf.nn.elu, data_format=data_format)


        # ## mask decoder
        # z_m = slim.fully_connected(z, z_num, activation_fn=None)
        #
        # num_output = int(np.prod([8, 8, hidden_num]))
        # x_m = slim.fully_connected(z_m, num_output, activation_fn=None)
        # x_m = reshape(x_m, 8, 8, hidden_num, data_format)
        #
        # for idx in range(repeat_num):
        #     x_m = slim.conv2d(x_m, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        #     x_m = slim.conv2d(x_m, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        #     if idx < repeat_num - 1:
        #         x_m = upscale(x_m, 2, data_format)
        #
        # maskout = slim.conv2d(x_m, 3, 3, 1, activation_fn=None, data_format=data_format)


        ## light estimator
        z_n = tf.transpose(z_n, [0, 3, 1, 2])
        z_a = tf.transpose(z_a, [0, 3, 1, 2])

        z_l = tf.concat([z, z_n, z_a], axis=1)
        # print z_l.get_shape() # 16 384 32 32
        x_l = slim.conv2d(z_l, 128, 1, 1, activation_fn=tf.nn.elu, data_format=data_format)
        # print x_l.get_shape() # 16 128 32 32
        x_l = tf.nn.avg_pool(x_l, ksize = [1, 1, 32, 32], strides = [1, 1, 1, 1], padding = 'VALID', data_format='NCHW')
        # print x_l.get_shape() # 16 128 1 1
        x_l = tf.reshape(x_l, [16, 128])
        # print x_l.get_shape() # 16 128
        lightout = slim.fully_connected(x_l, 27, activation_fn=None)
        shadingweight = slim.fully_connected(x_l, 3, activation_fn=None)
        # print lightout.get_shape() # 16 27
        # print shadingweight.get_shape() # 16 3
        # print lightout.dtype # float32


        ## reconstruction

        # shading = getshading10(normalout,lightout) # 16 3 64 64
        shading = getshading10(normalout,lightout, shadingweight) # 16 3 64 64

        recon = shading*albedoout

        ## relighting
        # pointlight 12*3 = light# * [lx ly lz]
        pointlight = tf.constant([
        [-1, 0, 0],
        [-1/np.sqrt(2.), 1/np.sqrt(2.), 0],
        [0, 1, 0],
        [1/np.sqrt(2.), 1/np.sqrt(2.), 0],
        [1, 0, 0],
        [-1/2., 1/np.sqrt(2.), 1/2.],
        [1/2., 1/np.sqrt(2.), 1/2.],
        [-1/np.sqrt(2.), 0, 1/np.sqrt(2.)],
        [0, 0, 1],
        [1/np.sqrt(2.), 0, 1/np.sqrt(2.)],
        [-1/2., -1/np.sqrt(2.), 1/2.],
        [1/2., -1/np.sqrt(2.), 1/2.]], dtype=tf.float32)

        # print (pointlight.get_shape()) # 12 3

        ## pointshading 16*64*64*12 = batch * x * y * light#
        pointshading = getpointshading(normalout, pointlight) # 16 64 64 12

        ## pointrecon 192(12*16)*3*64*64
        pointrecon = getpointrecon(albedoout, pointshading)


        # relight = np.random.normal(0, 1, size=(16, 27))
        # relight = tf.random_normal([16, 27], mean=0.0, stddev=1.0, dtype=tf.float32)
        # light2 = tf.cast(tf.reshape(tf.tile(tf.constant([1,1,1,1,1,1,1,1,1]),[48]),[16,27]),dtype=tf.float32)

        # relight = tf.random_shuffle(relight)
        # light2 = tf.random_shuffle(lightout)
        # print relight.get_shape() # 16 27



        # out = recon * maskout - maskout * bggt

    variables = tf.contrib.framework.get_variables(vs)
    return albedoout, normalout, maskout, lightout, shadingweight, shading, recon, pointrecon, variables
    # return albedoout, normalout, maskout, lightout, shadingweight, shading, recon, light2, shading2, recon2, variables
    # return normalout, maskout, albedoout, lightout, shading, recon, relight, reshading, recon2, variables
    # return normalout, albedoout, lightout, shading, recon, variables

def DiscriminatorCNN(x, input_channel, z_num, repeat_num, hidden_num, data_format):
    with tf.variable_scope("D") as vs:
        # Encoder
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)

        prev_channel_num = hidden_num
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            if idx < repeat_num - 1:
                x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
                #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')

        x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
        z = x = slim.fully_connected(x, z_num, activation_fn=None)

        # Decoder
        num_output = int(np.prod([8, 8, hidden_num]))
        x = slim.fully_connected(x, num_output, activation_fn=None)
        x = reshape(x, 8, 8, hidden_num, data_format)

        for idx in range(repeat_num):
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)

        out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None, data_format=data_format)

    # print out.get_shape() #32 3 64 64
    variables = tf.contrib.framework.get_variables(vs)
    return out, z, variables


def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def get_conv_shape(tensor, data_format):
    shape = int_shape(tensor)
    # always return [N, H, W, C]
    if data_format == 'NCHW':
        return [shape[0], shape[2], shape[3], shape[1]]
    elif data_format == 'NHWC':
        return shape

def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

def reshape(x, h, w, c, data_format):
    if data_format == 'NCHW':
        x = tf.reshape(x, [-1, c, h, w])
    else:
        x = tf.reshape(x, [-1, h, w, c])
    return x

def resize_nearest_neighbor(x, new_size, data_format):
    if data_format == 'NCHW':
        x = nchw_to_nhwc(x)
        x = tf.image.resize_nearest_neighbor(x, new_size)
        x = nhwc_to_nchw(x)
    else:
        x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def upscale(x, scale, data_format):
    _, h, w, _ = get_conv_shape(x, data_format)
    return resize_nearest_neighbor(x, (h*scale, w*scale), data_format)



###### from resnet.py

BN_EPSILON = 0.001

def activation_summary(x):
    '''
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''

    ## TODO: to allow different weight decay to fully connected layer and conv layer
    regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables


def output_layer(input_layer, num_labels):
    '''
    :param input_layer: 2D tensor
    :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
    :return: output layer Y = WX + B
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h


def batch_normalization_layer(input_layer, dimension):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer


def conv_bn_relu_layer(input_layer, filter_shape, stride):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    output = tf.nn.relu(bn_layer)
    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer



def residual_block(input_layer, output_channel, first_block=False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                     input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output


def normalresnet(input_tensor_batch, reuse):
    '''
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_residual_blocks
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    '''
    # input: 16 128 32 32
    layers = []
    # with tf.variable_scope('n_conv0', reuse=reuse):
    #     conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 16], 1)
    #     activation_summary(conv0)
    #     layers.append(conv0)

    with tf.variable_scope('n_conv1', reuse=reuse):
        conv1 = residual_block(input_tensor_batch, 128, first_block=True)
        activation_summary(conv1)
        layers.append(conv1)

    with tf.variable_scope('n_conv2', reuse=reuse):
        conv2 = residual_block(layers[-1], 128)
        activation_summary(conv2)
        layers.append(conv2)

    with tf.variable_scope('n_conv3', reuse=reuse):
        conv3 = residual_block(layers[-1], 128)
        activation_summary(conv3)
        layers.append(conv3)

    with tf.variable_scope('n_conv4', reuse=reuse):
        conv4 = residual_block(layers[-1], 128)
        activation_summary(conv4)
        layers.append(conv4)

    with tf.variable_scope('n_conv5', reuse=reuse):
        conv5 = residual_block(layers[-1], 128)
        activation_summary(conv5)
        layers.append(conv5)
    assert conv5.get_shape().as_list()[1:] == [32, 32, 128]


    with tf.variable_scope('n_out', reuse=reuse):
        in_channel = layers[-1].get_shape().as_list()[-1]
        bn_layer = batch_normalization_layer(layers[-1], in_channel)
        output = tf.nn.relu(bn_layer)

    return output





def albedoresnet(input_tensor_batch, n, reuse):
    '''
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_residual_blocks
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    '''

    layers = []

    with tf.variable_scope('a_conv1', reuse=reuse):
        conv1 = residual_block(input_tensor_batch, 128, first_block=True)
        activation_summary(conv1)
        layers.append(conv1)

    with tf.variable_scope('a_conv2', reuse=reuse):
        conv2 = residual_block(layers[-1], 128)
        activation_summary(conv2)
        layers.append(conv2)

    with tf.variable_scope('a_conv3', reuse=reuse):
        conv3 = residual_block(layers[-1], 128)
        activation_summary(conv3)
        layers.append(conv3)

    with tf.variable_scope('a_conv4', reuse=reuse):
        conv4 = residual_block(layers[-1], 128)
        activation_summary(conv4)
        layers.append(conv4)

    with tf.variable_scope('a_conv5', reuse=reuse):
        conv5 = residual_block(layers[-1], 128)
        activation_summary(conv5)
        layers.append(conv5)
    assert conv5.get_shape().as_list()[1:] == [32, 32, 128]

    with tf.variable_scope('a_out', reuse=reuse):
        in_channel = layers[-1].get_shape().as_list()[-1]
        bn_layer = batch_normalization_layer(layers[-1], in_channel)
        output = tf.nn.relu(bn_layer)

    return output

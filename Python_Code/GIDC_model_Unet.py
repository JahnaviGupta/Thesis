# -*- coding: utf-8 -*-
"""
By Fei Wang, Jan 1, 2022
Contact: WangFei_m@outlook.com
This code implements the DNN structure and measurements process of GIDC algorithm reported in the paper: 
Fei Wang et al. 'Far-field super-resolution ghost imaging with adeep neural network constraint'. Light Sci Appl 11, 1 (2022).  
https://doi.org/10.1038/s41377-021-00680-w
Please cite our paper if you find this code offers any help.

Inputs:
inpt: DGI results (batch_size * pixels * pixels * 1)
real_A: illumination patterns (batch_size * pixels * pixels * num_patterns)
batch_size: batch_size
img_W: width of image
img_H: high of image
num_A: num_patterns

Outputs:
out_x: estimated image by GIDC (batch_size * pixels * pixels * 1)
out_y: estimated intensity measurements associated with out_x and real_A (batch_size * 1 * 1 * num_patterns)
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU

def weight_variable(shape, name=None):
    initial = tf.random.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)
    
def leaky_relu(x, leak=0.2, name=''):
    return tf.maximum(x, x * leak, name=name)

def inference(inpt, real_A, batch_size, img_W, img_H, num_A, isTrain=True):
    c_size = 5
    d_size = 5

    layer_1 = tf.reshape(inpt, [batch_size, img_W, img_H, 1])

    with tf.compat.v1.variable_scope('conv0'):
        conv0 = Conv2D(16, (c_size, c_size), padding='same')(layer_1)
        conv0 = BatchNormalization()(conv0, training=isTrain)
        conv0 = LeakyReLU()(conv0)

    with tf.compat.v1.variable_scope('conv1'):
        conv1 = Conv2D(16, (c_size, c_size), padding='same')(conv0)
        conv1 = BatchNormalization()(conv1, training=isTrain)
        conv1 = LeakyReLU()(conv1)

    with tf.compat.v1.variable_scope('conv_pooling_1'):
        Convpool_1 = Conv2D(16, (d_size, d_size), strides=(2, 2), padding='same')(conv1)
        Convpool_1 = LeakyReLU()(BatchNormalization()(Convpool_1, training=isTrain))

    with tf.compat.v1.variable_scope('conv2'):
        conv2 = Conv2D(32, (c_size, c_size), padding='same')(Convpool_1)
        conv2 = BatchNormalization()(conv2, training=isTrain)
        conv2 = LeakyReLU()(conv2)

    with tf.compat.v1.variable_scope('conv_pooling_2'):
        Convpool_2 = Conv2D(32, (d_size, d_size), strides=(2, 2), padding='same')(conv2)
        Convpool_2 = LeakyReLU()(BatchNormalization()(Convpool_2, training=isTrain))

    with tf.compat.v1.variable_scope('conv3'):
        conv3 = Conv2D(64, (c_size, c_size), padding='same')(Convpool_2)
        conv3 = BatchNormalization()(conv3, training=isTrain)
        conv3 = LeakyReLU()(conv3)

    with tf.compat.v1.variable_scope('conv_pooling_3'):
        Convpool_3 = Conv2D(64, (d_size, d_size), strides=(2, 2), padding='same')(conv3)
        Convpool_3 = LeakyReLU()(BatchNormalization()(Convpool_3, training=isTrain))

    with tf.compat.v1.variable_scope('conv4'):
        conv4 = Conv2D(128, (c_size, c_size), padding='same')(Convpool_3)
        conv4 = BatchNormalization()(conv4, training=isTrain)
        conv4 = LeakyReLU()(conv4)

    with tf.compat.v1.variable_scope('conv_pooling_4'):
        Convpool_4 = Conv2D(128, (d_size, d_size), strides=(2, 2), padding='same')(conv4)
        Convpool_4 = LeakyReLU()(BatchNormalization()(Convpool_4, training=isTrain))

    with tf.compat.v1.variable_scope('conv5'):
        conv5 = Conv2D(256, (c_size, c_size), padding='same')(Convpool_4)
        conv5 = BatchNormalization()(conv5, training=isTrain)
        conv5 = LeakyReLU()(conv5)

    # Upsampling to match output dimensions
    with tf.compat.v1.variable_scope('conv_transpose1'):
        conv6 = Conv2DTranspose(128, (d_size, d_size), strides=(2, 2), padding='same')(conv5)
        conv6 = BatchNormalization()(conv6, training=isTrain)
        conv6 = LeakyReLU()(conv6)

    with tf.compat.v1.variable_scope('conv_transpose2'):
        conv7 = Conv2DTranspose(64, (d_size, d_size), strides=(2, 2), padding='same')(conv6)
        conv7 = BatchNormalization()(conv7, training=isTrain)
        conv7 = LeakyReLU()(conv7)

    with tf.compat.v1.variable_scope('conv_transpose3'):
        conv8 = Conv2DTranspose(32, (d_size, d_size), strides=(2, 2), padding='same')(conv7)
        conv8 = BatchNormalization()(conv8, training=isTrain)
        conv8 = LeakyReLU()(conv8)

    with tf.compat.v1.variable_scope('conv_transpose4'):
        conv9 = Conv2DTranspose(16, (d_size, d_size), strides=(2, 2), padding='same')(conv8)
        conv9 = BatchNormalization()(conv9, training=isTrain)
        conv9 = LeakyReLU()(conv9)

    with tf.compat.v1.variable_scope('conv10'):
        conv10 = Conv2D(1, (c_size, c_size), padding='same')(conv9)
        conv10 = BatchNormalization()(conv10, training=isTrain)
        conv10 = tf.nn.sigmoid(conv10)

    # the measurement process of ghost imaging (physical model)
    with tf.compat.v1.variable_scope('measurement'):
        out_x = tf.reshape(conv10, [batch_size, img_W, img_H, 1])
        out_x = out_x / tf.reduce_max(out_x)

        pattern = tf.reshape(real_A, [img_W, img_H, 1, num_A])
        out_y = tf.nn.conv2d(out_x, pattern, strides=[1, 1, 1, 1], padding='VALID')

        mean_x, variance_x = tf.nn.moments(out_x, [0, 1, 2, 3])
        mean_y, variance_y = tf.nn.moments(out_y, [0, 1, 2, 3])
        out_x = (out_x - mean_x) / tf.sqrt(variance_x)
        out_y = (out_y - mean_y) / tf.sqrt(variance_y)

    return out_x, out_y
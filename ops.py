import tensorflow as tf
import numpy as np

def xavier_stddev(size):
    in_dim = size[0]
    return (1. / tf.sqrt(in_dim/ 2.))

def lrelu(x, leak = 0.2, name ='lrelu'):
    return tf.maximum(x, leak*x)

def conv2d(input_, output_dim, f_h = 4, f_w = 4, s_h = 2, s_w = 2, name = 'conv2d'):
    size = [f_h, f_w,input_.get_shape()[-1], output_dim]
    stddev = xavier_stddev(size)
    with tf.variable_scope(name):
        w = tf.get_variable('w', size, initializer = tf.truncated_normal_initializer(stddev = stddev))
        conv = tf.nn.conv2d(input_, w, strides = [1, s_h, s_w, 1], padding = 'SAME')
        bias = tf.get_variable('bias', [output_dim], initializer = tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, bias), conv.get_shape())

    return conv

def conv3d(input_, output_dim, f_d = 4, f_h = 4, f_w = 4, s_d = 2, s_h = 2, s_w =2, name = 'conv3d', padding = 'SAME'):
    size = [f_d,f_h,f_w, input_.get_shape()[-1], output_dim]
    stddev = xavier_stddev(size)
    with tf.variable_scope(name):
        w = tf.get_variable('w', size, initializer = tf.truncated_normal_initializer(stddev = stddev))
        conv = tf.nn.conv3d(input_, w, strides = [1, s_d, s_h, s_w, 1], padding = 'SAME')
        bias = tf.get_variable('bias', [output_dim], initializer = tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, bias), conv.get_shape())

    return conv

def deconv2d(input_, output_shape, f_h = 4, f_w = 4, s_h =2, s_w = 2, name = 'deconv2d'):
    size = [f_h, f_w, output_shape[-1], input_.get_shape()[-1]]
    stddev = xavier_stddev(size)
    with tf.variable_scope(name):
        w = tf.get_variable('w', size, initializer = tf.truncated_normal_initializer(stddev = stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape = output_shape, strides = [1,s_h,s_w,1], padding = 'SAME')
        bias = tf.get_variable('bias', [output_shape[-1]], initializer = tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv,bias), deconv.get_shape())
    return deconv

def deconv3d(input_, output_shape, f_d =4, f_h =4, f_w =4, s_d =2, s_h =2, s_w =2, name = 'deconv3d'):
    size = [f_d,f_h,f_w, output_shape[-1], input_.get_shape()[-1]]
    stddev = xavier_stddev(size)
    with tf.variable_scope(name):
        w = tf.get_variable('w', size, initializer = tf.truncated_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv3d_transpose(input_, w, output_shape = output_shape, strides = [1, s_d, s_h, s_w, 1], padding ='SAME')
        bias = tf.get_variable('bias', [output_shape[-1]], initializer = tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv,bias), deconv.get_shape())
    return deconv

def batch_norm(x, train = True):
    epsilon = 1e-5
    momentum = 0.9
    name = 'batch_norm'
    return tf.contrib.layers.batch_norm(x,
            decay = momentum,
            updates_collections = None,
            epsilon = epsilon,
            scale = True,
            is_training = train)


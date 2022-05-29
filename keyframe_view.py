#coding:utf-8
#!usr/bin/env python
#-*- coding:utf-8 _*-


import numpy as np
import tensorflow as tf
import ops


def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1,name=name)
    return tf.Variable(initial)


# bais initialization
def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape,name=name)
    return tf.Variable(initial)


def conv2d_function(input,filter_shape,b_shape,strides_shape,name):
    w = weight_variable(filter_shape,name)
    b = bias_variable(b_shape,name)
    conv2d_out = tf.nn.leaky_relu(
        tf.nn.conv2d(input, w, strides=strides_shape, padding='SAME') + b)
    print('conv2d'+name, conv2d_out.shape)
    return conv2d_out


def dilated_conv2d( inputs, w1, rate, b1):
    dilated_conv1 = tf.nn.atrous_conv2d(value=inputs, filters=w1, rate=rate, padding='SAME') + b1
    return dilated_conv1



def sn_conv1x1(input_, output_dim, update_collection,
              init=tf.contrib.layers.xavier_initializer(), name='sn_conv1x1'):
  with tf.variable_scope(name):
    k_h = 1
    k_w = 1
    d_h = 1
    d_w = 1
    w = tf.get_variable(
        'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
        initializer=init)
    w_bar = ops.spectral_normed_weight(w, num_iters=1, update_collection=update_collection)

    conv = tf.nn.conv2d(input_, w_bar, strides=[1, d_h, d_w, 1], padding='SAME')
    return conv

def HDC_residual_block(x, num_input_filters, num_output_filters, hdc_block_num):

    # implementing residual block logic
    input = tf.contrib.layers.batch_norm(x)
    input = tf.nn.relu(input)
    O=conv2d_function(input=input,filter_shape=[3,3,num_input_filters,num_output_filters],b_shape=[num_output_filters],strides_shape=[1,2,2,1],name='conv_stride'+ str(hdc_block_num))


    intermediate = tf.contrib.layers.batch_norm(O)
    intermediate = tf.nn.relu(intermediate)
    intermediate = sn_conv1x1(intermediate, intermediate.get_shape()[-1], update_collection=None,
                              name='conv1x1' + str(hdc_block_num))


    w_conv_1 = tf.get_variable('hdc_rs_block_' + str(hdc_block_num) + '_w_conv_1',
                               shape=[3, 3, num_output_filters, num_output_filters], dtype=tf.float32)
    b_conv_1 = tf.get_variable('hdc_rs_block_' + str(hdc_block_num) + '_b_conv_1', shape=[num_output_filters], dtype=tf.float32)
    weight_layer_1=dilated_conv2d(intermediate,w_conv_1,1,b_conv_1)


    if num_input_filters != num_output_filters:
        w_conv_increase = tf.get_variable('hdc_rs_block_' + str(hdc_block_num) + '_w_conv_increase',
                                          shape=[1, 1, num_output_filters, num_output_filters], dtype=tf.float32)
        b_conv_increase = tf.get_variable('hdc_rs_block_' + str(hdc_block_num) + '_b_conv_increase', shape=[num_output_filters],
                                          dtype=tf.float32)

        intermediate=dilated_conv2d(intermediate,w_conv_increase,1,b_conv_increase)
        #x = tf.nn.conv2d(x, w_conv_increase, strides=[1, 1, 1, 1], padding='SAME') + b_conv_increase
    O1 = tf.add(intermediate, weight_layer_1)

    intermediate1 = tf.contrib.layers.batch_norm(O1)
    intermediate1 = tf.nn.relu(intermediate1)
    intermediate1 = sn_conv1x1(intermediate1, intermediate1.get_shape()[-1], update_collection=None,
                              name='conv1x1_1' + str(hdc_block_num))

    w_conv_2 = tf.get_variable('hdc_rs_block_' + str(hdc_block_num) + '_w_conv_2',
                               shape=[3, 3, num_output_filters, num_output_filters], dtype=tf.float32)
    b_conv_2 = tf.get_variable('hdc_rs_block_' + str(hdc_block_num) + '_b_conv_2', shape=[num_output_filters], dtype=tf.float32)
    weight_layer_2 = dilated_conv2d(intermediate1, w_conv_2, 2, b_conv_2)


    intermediate2 = tf.contrib.layers.batch_norm(weight_layer_2)
    intermediate2 = tf.nn.relu(intermediate2)
    intermediate2 = sn_conv1x1(intermediate2, intermediate2.get_shape()[-1], update_collection=None,
                              name='conv1x1_2' + str(hdc_block_num))

    w_conv_3 = tf.get_variable('hdc_rs_block_' + str(hdc_block_num) + '_w_conv_3',
                               shape=[3, 3, num_output_filters, num_output_filters], dtype=tf.float32)
    b_conv_3 = tf.get_variable('hdc_rs_block_' + str(hdc_block_num) + '_b_conv_3', shape=[num_output_filters], dtype=tf.float32)

    weight_layer_3 = dilated_conv2d(intermediate2, w_conv_3, 3, b_conv_3)

    O2 = tf.add(O1 , weight_layer_3)


    print(str(hdc_block_num)+':',O2.shape)

    return O2

def global_average_pooling(feature_map,k_size,stride,name):
    h = tf.nn.avg_pool(feature_map, ksize=k_size, strides=stride, padding='VALID',name=name)
    h = tf.reduce_mean(h, axis=[1, 2],name=name)
    print(name, h.shape)
    return h


def get_super_resolved_features(x):

    hdc_rs_block_1 = HDC_residual_block(x, 1, 32, 1)

    gap1=global_average_pooling(hdc_rs_block_1, [1,2,2,1], [1,1,1,1],'gap1')

    hdc_rs_block_2 = HDC_residual_block(hdc_rs_block_1, 32, 64, 2)

    gap2 = global_average_pooling(hdc_rs_block_2, [1, 2, 2, 1], [1, 1, 1, 1], 'gap2')

    hdc_rs_block_3 = HDC_residual_block(hdc_rs_block_2, 64, 128, 3)

    gap3 = global_average_pooling(hdc_rs_block_3, [1, 2, 2, 1], [1, 1, 1, 1], 'gap3')

    hdc_rs_block_4 = HDC_residual_block(hdc_rs_block_3, 128, 256, 4)

    gap4 = global_average_pooling(hdc_rs_block_4, [1, 2, 2, 1], [1, 1, 1, 1], 'gap4')

    hdc_rs_block_5 = HDC_residual_block(hdc_rs_block_4, 256, 512, 5)

    gap5 = global_average_pooling(hdc_rs_block_5, [1, 2, 2, 1], [1, 1, 1, 1], 'gap5')

    hdc_rs_block_6 = HDC_residual_block(hdc_rs_block_5, 512, 512, 6)

    gap6 = global_average_pooling(hdc_rs_block_6, [1, 2, 2, 1], [1, 1, 1, 1], 'gap6')

    gap_features=[gap1,gap2,gap3,gap4,gap5,gap6]
    concat_feature=tf.concat(gap_features,axis=1)


    return concat_feature



#coding:utf-8
#!usr/bin/env python
#-*- coding:utf-8 _*-


import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.python.ops.metrics_impl import mean_absolute_error
#import res_block
from tensorflow.contrib import layers
import ops
from numpy import *
import math
import keyframe_view


def weight_variable(shape,name_w):
    intial=tf.truncated_normal(shape=shape,stddev=0.1,name=name_w)
    return tf.Variable(intial)

# bais initialization
def bias_variable(shape,name_b):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv3d_function(input,filter_shape,b_shape,strides_shape,name,norm):
    w = weight_variable(filter_shape,name)
    b = bias_variable(b_shape,name)

    Wx_plus_b=tf.nn.conv3d(input, w, strides=strides_shape, padding='SAME') + b

    if norm:  # 判断书否是 BN 层
        fc_mean, fc_var = tf.nn.moments(
            Wx_plus_b,
            axes=[0,1,2,3],

        )
        scale = tf.Variable(tf.ones(b_shape))
        shift = tf.Variable(tf.zeros(b_shape))
        epsilon = 0.001
        Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, fc_mean, fc_var, shift, scale, epsilon)
        # 上面那一步, 在做如下事情:

    conv3d_out = tf.nn.leaky_relu(Wx_plus_b)
    return conv3d_out

def conv2d_function(input,filter_shape,b_shape,strides_shape,name):
    w = weight_variable(filter_shape,name)
    b = bias_variable(b_shape,name)
    conv2d_out = tf.nn.leaky_relu(
        tf.nn.conv2d(input, w, strides=strides_shape, padding='SAME') + b)
    return conv2d_out



class mv_qca_heal_model(object):
    def __init__(self,l2_reg_lambda=0.00001):

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.l2_reg_lambda = l2_reg_lambda
        #self.batch_size=batch_size

        self.input_x1 = tf.placeholder(tf.float32, [None, 10, 256, 256, 1])
        self.input_x2 = tf.placeholder(tf.float32, [None, 10, 256, 256, 1])
        self.mainkey_input = tf.placeholder(tf.float32, [None, 256, 256, 1])
        self.input_y1 = tf.placeholder(tf.float32, [None, 6])

        self.main_keyframe_feature1 = keyframe_view.get_super_resolved_features(self.mainkey_input)

        # print('keyframe_fea.shape',keyframe_fea.shape)
        with tf.name_scope('main_keyframe'):
            # main_frame_feature2 = tf.reshape(self.main_keyframe_feature1, [-1, 8192])
            # print(main_frame_feature2.shape)

            w1_frame = weight_variable([1504, 512],'keyframe_w')
            b1_frame = bias_variable([512],'keyframe_b')
            self.main_frame_feature2 = tf.nn.leaky_relu(tf.matmul(self.main_keyframe_feature1, w1_frame) + b1_frame)
            print(self.main_frame_feature2.shape)


        ##########################
        #       LAO attention
        ##########################

        conv3d_lao_out=self.viewpoint_conv3d(self.input_x1,'lao')
        lao_self_att_out=sn_3d_non_local_block_sim(conv3d_lao_out,None,'lao')#(?, 10, 4, 4, 512)
        lao_context_att_input1=tf.reshape(lao_self_att_out,[-1,lao_self_att_out.shape[1],lao_self_att_out.shape[2]*lao_self_att_out.shape[3],lao_self_att_out.shape[4]])
        lao_context_att_out1=self.AttentionLayer(lao_context_att_input1,'lao_att1')
        lao_context_att_input2=tf.expand_dims(lao_context_att_out1,1)

        lao_context_att_out2 = self.AttentionLayer(lao_context_att_input2, 'lao_att2')

        self.lao_feature=tf.reshape(lao_context_att_out2,[-1,lao_context_att_out2.get_shape()[-1]])


        ##########################
        #       LAO attention
        ##########################

        conv3d_rao_out = self.viewpoint_conv3d(self.input_x2, 'rao')
        # print(conv3d_rao_out[1])
        # print(conv3d_rao_out.shape)
        rao_self_att_out = sn_3d_non_local_block_sim(conv3d_rao_out, None, 'rao')  # (?, 10, 4, 4, 512)
        rao_context_att_input1 = tf.reshape(rao_self_att_out, [-1, rao_self_att_out.shape[1],
                                                               rao_self_att_out.shape[2] * rao_self_att_out.shape[3],
                                                               rao_self_att_out.shape[4]])

        rao_context_att_out1 = self.AttentionLayer(rao_context_att_input1, 'rao_att1')

        rao_context_att_input2 = tf.expand_dims(rao_context_att_out1, 1)

        rao_context_att_out2 = self.AttentionLayer(rao_context_att_input2, 'rao_att2')

        self.rao_feature = tf.reshape(rao_context_att_out2, [-1, rao_context_att_out2.get_shape()[-1]])



        #######################
        #       loss
        #######################
        with tf.name_scope('hierarchical_shared_layer'):
            with tf.name_scope('view_share'):
                discrimination_key=tf.sigmoid(tf.log(tf.abs(self.main_frame_feature2)))
                discrimination_lao_view=tf.sigmoid(tf.log(tf.abs(self.lao_feature)))
                discrimination_rao_view = tf.sigmoid(tf.log(tf.abs(self.rao_feature)))
                key_specific=tf.multiply(discrimination_key,self.main_frame_feature2)
                lao_specific=tf.multiply(discrimination_lao_view,self.lao_feature)
                rao_specific=tf.multiply(discrimination_rao_view,self.rao_feature)

                self.qca_feature=tf.concat([key_specific,lao_specific,rao_specific],axis=1)




        with tf.name_scope('task_full_connection_reg'):


            w0_reg_qca = weight_variable([1536, 512],'w0_reg_qca')
            b0_reg_qca = bias_variable([512],'b0_reg_qca')
            f0_reg_qca = tf.nn.leaky_relu(tf.matmul(self.qca_feature, w0_reg_qca) + b0_reg_qca)
            f0_reg_qca=tf.nn.dropout(f0_reg_qca, self.dropout_keep_prob)


            self.qca_concat_feature=tf.concat([f0_reg_qca],axis=1)

            self.w_out=weight_variable([512,6],name_w='w_out')
            b_out=bias_variable([6],name_b='b_out')
            self.f1_reg_qca=tf.nn.leaky_relu(tf.matmul(self.qca_concat_feature, self.w_out) + b_out)



            with tf.name_scope("loss"):
                self.loss_reg_qca = tf.reduce_mean(tf.abs(self.input_y1 - self.f1_reg_qca))

                self.consistency_loss=tf.reduce_mean(tf.square(self.rao_feature-self.lao_feature))

                self.l2_losses = tf.add_n(
                    [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'b' not in v.name]) * l2_reg_lambda

                self.losses = self.loss_reg_qca + self.l2_losses+0.01*self.consistency_loss


            with tf.name_scope('performance'):
                self.mae_qca_ = tf.abs(self.input_y1 - self.f1_reg_qca, name='mae_qca_')
                self.mae_qca = tf.reduce_mean(self.mae_qca_)

    def abs_smooth(self,x):
        """Smoothed absolute function. Useful to compute an L1 smooth error.
        Define as:
            x^2 / 2         if abs(x) < 1
            abs(x) - 0.5    if abs(x) > 1
        We use here a differentiable definition using min(x) and abs(x). Clearly
        not optimal, but good enough for our purpose!
        """
        absx = tf.abs(x)
        minx = tf.minimum(absx, 1)
        r = 0.5 * ((absx - 1) * minx + absx)
        return r




    def viewpoint_conv3d(self,input,view_name):
        conv3d1_view = conv3d_function(input, filter_shape=[2, 3, 3, 1, 64], b_shape=[64],
                                      strides_shape=[1, 1, 2, 2, 1], name=view_name+'3d1',norm=True)
        conv3d2_view = conv3d_function(conv3d1_view, filter_shape=[2, 3, 3, 64, 128], b_shape=[128],
                                       strides_shape=[1, 1, 2, 2, 1], name=view_name + '3d2',norm=True)
        conv3d3_view = conv3d_function(conv3d2_view, filter_shape=[2, 3, 3, 128, 128], b_shape=[128],
                                       strides_shape=[1, 1, 2, 2, 1], name=view_name + '3d3',norm=True)
        conv3d4_view = conv3d_function(conv3d3_view, filter_shape=[2, 2, 2, 128, 256], b_shape=[256],
                                       strides_shape=[1, 1, 2, 2, 1], name=view_name + '3d4',norm=True)
        conv3d5_view = conv3d_function(conv3d4_view, filter_shape=[2, 2, 2, 256, 256], b_shape=[256],
                                       strides_shape=[1, 1, 2, 2, 1], name=view_name + '3d5',norm=True)
        conv2d1_view=conv3d_function(conv3d5_view, filter_shape=[1, 2, 2, 256, 512], b_shape=[512],strides_shape=[1, 1, 2, 2, 1], name=view_name+'2d1',norm=True)

        return conv2d1_view




    def AttentionLayer(self, inputs, name):
        with tf.variable_scope(name):
            u_context = tf.Variable(tf.truncated_normal(shape=[int(inputs.shape[-1])]), name='u_context')
            h = layers.fully_connected(inputs=inputs, num_outputs=int(inputs.shape[-1]),activation_fn=tf.nn.tanh)
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=3, keep_dims=True), dim=2)
            attention_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=2)
            return attention_output



def dilated_conv2d( inputs, w_shape, rate, b_shape,name_w,name_b):
    w1 = weight_variable(w_shape,name_w)
    b1 = bias_variable(b_shape,name_b)
    dilated_conv1 = tf.nn.atrous_conv2d(value=inputs, filters=w1, rate=rate, padding='SAME') + b1
    dilated_conv1_relu = tf.nn.relu(dilated_conv1)
    return dilated_conv1_relu


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

def sn_3dconv1x1(input_, output_dim, update_collection,
             init=tf.contrib.layers.xavier_initializer(), name='sn_3dconv1x1'):
  with tf.variable_scope(name):

      k_h = 1
      k_w = 1
      d_h = 1
      d_w = 1
      w = tf.get_variable(
          'w', [1,k_h, k_w, input_.get_shape()[-1], output_dim],
          initializer=init)
      w_bar = ops.spectral_normed_weight(w, num_iters=1, update_collection=update_collection)

      conv = tf.nn.conv3d(input_, w_bar, strides=[1,1, d_h, d_w, 1], padding='SAME')
      return conv

def sn_3d_non_local_block_sim(x, update_collection, name, init=tf.contrib.layers.xavier_initializer()):
  with tf.variable_scope(name):
    x_shape=x.get_shape()
    b,h, w, num_channels=x_shape[-4],x_shape[-3],x_shape[-2],x_shape[-1]

    #batch_size, h, w, num_channels = x.get_shape().as_list()
    location_num = h * w
    downsampled_num = location_num // 4

    # theta path
    theta = sn_3dconv1x1(x, num_channels // 8, update_collection, init, 'sn_conv_theta')
    theta = tf.reshape(
        theta, [-1,b, location_num, num_channels // 8])

    # phi path
    phi = sn_3dconv1x1(x, num_channels // 8, update_collection, init, 'sn_conv_phi')
    phi = tf.layers.max_pooling3d(inputs=phi, pool_size=[1,2, 2], strides=[1,2,2],padding='SAME')
    phi = tf.reshape(
        phi, [-1,b, downsampled_num, num_channels // 8])

    attn = tf.matmul(theta, phi, transpose_b=True)
    attn = tf.nn.softmax(attn)
    print(tf.reduce_sum(attn, axis=-1))

    # g path
    g = sn_3dconv1x1(x, num_channels // 2, update_collection, init, 'sn_conv_g')
    g = tf.layers.max_pooling3d(inputs=g, pool_size=[1,2, 2], strides=[1,2,2],padding='SAME')
    g = tf.reshape(
      g, [-1,b, downsampled_num, num_channels // 2])

    attn_g = tf.matmul(attn, g)
    attn_g = tf.reshape(attn_g, [-1,b, h, w, num_channels // 2])
    sigma = tf.get_variable(
        'sigma_ratio', [], initializer=tf.constant_initializer(0.0))
    attn_g = sn_3dconv1x1(attn_g, num_channels, update_collection, init, 'sn_conv_attn')
    o=x + sigma * attn_g
    print('o.shape',o.shape)
    return o



model=mv_qca_heal_model()

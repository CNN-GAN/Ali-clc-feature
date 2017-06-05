from __future__ import division
import tensorflow as tf
from ops import *
from utils import *


def discA(image, options, reuse=False, name="discA"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, s=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = conv2d(h3, 1, s=1, name='d_h3_pred')
        # h4 is (32 x 32 x 1)
        return h4

def discB(code, options, reuse=False, name="discB"):

    with tf.variable_scope(name):
        # code is 512
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        fc1 = dense(code, options.df_dim*16, name='fc1')
        # fc1 is (df_dim*8)
        fc2 = dense(fc1,  options.df_dim*8, name='fc2')        
        # fc2 is (df_dim*8)

        return fc2

def discAB(image, code, options, reuse=False, name="discriminatorAB"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        # code is 512
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        m1_h0 = lrelu(conv2d(image, options.df_dim, name='m1_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        m1_h1 = lrelu(instance_norm(conv2d(m1_h0, options.df_dim*2, name='m1_h1_conv'), 'm1_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        m1_h2 = lrelu(instance_norm(conv2d(m1_h1, options.df_dim*4, name='m1_h2_conv'), 'm1_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        m1_h3 = lrelu(instance_norm(conv2d(m1_h2, options.df_dim*8, s=1, name='m1_h3_conv'), 'm1_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
        m1_h4 = conv2d(m1_h3, 1, s=1, name='m1_h4')
        # h4 is (32 x 32 x 1)
        m1_h5 = dense(flatten(m1_h4, name='m1_flatten'), options.df_dim*8, name='m1_h5')
        # h5 is (512)

        m2_fc1 = dense(code, options.df_dim*16, name='m2_fc1')
        m2_fc2 = dense(m2_fc1,  options.df_dim*8,  name='m2_fc2')
        
        print (tf.shape(m2_fc2))
        m_h0  = tf.concat([m1_h5, m2_fc2], 1)
        m_h1  = dense(m_h0, options.df_dim*16, name='m_h1')
        m_out = dense(m_h1, options.df_dim*16, name='m_h2')

        return m_out

def encoder_unet(image, options, reuse=False, name='encoder'):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        # image is (256 x 256 x input_c_dim)
        e1 = conv2d(image, options.gf_dim, name='e1_conv')
        # e1 is (128 x 128 x self.gf_dim)
        e2 = instance_norm(conv2d(lrelu(e1), options.gf_dim*2,  name='e2_conv'), 'e2_bn')
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = instance_norm(conv2d(lrelu(e2), options.gf_dim*4,  name='e3_conv'), 'e3_bn')
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = instance_norm(conv2d(lrelu(e3), options.gf_dim*8,  name='e4_conv'), 'e4_bn')
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = instance_norm(conv2d(lrelu(e4), options.gf_dim*8,  name='e5_conv'), 'e5_bn')
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = instance_norm(conv2d(lrelu(e5), options.gf_dim*8,  name='e6_conv'), 'e6_bn')
        # e6 is (4 x 4 x self.gf_dim*8)
        e7 = instance_norm(conv2d(lrelu(e6), options.gf_dim*8,  name='e7_conv'), 'e7_bn')
        # e7 is (2 x 2 x self.gf_dim*8)
        e8 = instance_norm(conv2d(lrelu(e7), options.gf_dim*16, name='e8_conv'), 'e8_bn')
        # e8 is (1 x 1 x self.gf_dim*16)

        return tf.nn.relu(flatten(e8, name='e8_flatten'))

def decoder_unet(code, options, reuse=False, name='decoder'):

    with tf.variable_scope(name):
        # code is 512
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        d0 = tf.reshape(code, [-1, 1, 1, 512])

        d1 = deconv2d(d0, options.gf_dim*8, name='g_d1')
        d1 = tf.nn.dropout(instance_norm(d1, 'g_bn_d1'), 0.5)
        # d1 is (2 x 2 x self.gf_dim*8)

        d2 = deconv2d(tf.nn.relu(d1), options.gf_dim*8, name='g_d2')
        d2 = tf.nn.dropout(instance_norm(d2, 'g_bn_d2'), 0.5)
        # d2 is (4 x 4 x self.gf_dim*8)

        d3 = deconv2d(tf.nn.relu(d2), options.gf_dim*8, name='g_d3')
        d3 = tf.nn.dropout(instance_norm(d3, 'g_bn_d3'), 0.5)
        # d3 is (8 x 8 x self.gf_dim*8)

        d4 = deconv2d(tf.nn.relu(d3), options.gf_dim*8, name='g_d4')
        d4 = tf.nn.dropout(instance_norm(d4, 'g_bn_d4'), 0.5)
        # d4 is (16 x 16 x self.gf_dim*8)

        d5 = deconv2d(tf.nn.relu(d4), options.gf_dim*4, name='g_d5')
        d5 = tf.nn.dropout(instance_norm(d5, 'g_bn_d5'), 0.5)
        # d5 is (32 x 32 x self.gf_dim*4)

        d6 = deconv2d(tf.nn.relu(d5), options.gf_dim*2, name='g_d6')
        d6 = instance_norm(d6, 'g_bn_d6')
        # d6 is (64 x 64 x self.gf_dim*2)

        d7 = deconv2d(tf.nn.relu(d6), options.gf_dim, name='g_d7')
        d7 = instance_norm(d7, 'g_bn_d7')
        # d7 is (128 x 128 x self.gf_dim*1)

        d8 = deconv2d(tf.nn.relu(d7), options.output_c_dim, name='g_d8')
        # d8 is (256 x 256 x output_c_dim)
        
        return tf.nn.tanh(d8)

def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))

def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)

def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

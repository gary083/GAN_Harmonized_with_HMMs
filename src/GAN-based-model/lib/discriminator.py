import tensorflow as tf
import numpy as np

def weak_discriminator(inputs, embedding_matrix, first_channels, second_channels, reuse=False):
    #two layers with second layer very few channels
    with tf.variable_scope("weak_discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        embedding_matrix = tf.tile(tf.expand_dims(embedding_matrix, axis=0), [tf.shape(inputs)[0],1,1])
        inputs = tf.matmul(inputs, embedding_matrix)

        conv_3_1 = tf.layers.conv1d(inputs, first_channels, 3, padding='same', name='conv_3_1')
        conv_5_1 = tf.layers.conv1d(inputs, first_channels, 5, padding='same', name='conv_5_1')
        conv_7_1 = tf.layers.conv1d(inputs, first_channels, 7, padding='same', name='conv_7_1')
        conv_9_1 = tf.layers.conv1d(inputs, first_channels, 9, padding='same', name='conv_9_1')
        outputs = tf.concat([conv_3_1, conv_5_1, conv_7_1, conv_9_1], axis=-1)
        outputs = lrelu(outputs)
        conv_3_2 = tf.layers.conv1d(outputs, second_channels, 3, padding='same', name='conv_3_2')
        conv_5_2 = tf.layers.conv1d(outputs, second_channels, 3, padding='same', name='conv_5_2')
        conv_7_2 = tf.layers.conv1d(outputs, second_channels, 3, padding='same', name='conv_7_2')
        conv_9_2 = tf.layers.conv1d(outputs, second_channels, 3, padding='same', name='conv_9_2')
        outputs = tf.concat([conv_3_2, conv_5_2, conv_7_2, conv_9_2], axis=-1)
        outputs = tf.reshape(outputs, [-1, tf.cast(inputs.get_shape()[1]*second_channels*4, tf.int32)])  
        outputs = lrelu(outputs)
        outputs = tf.layers.dense(outputs, 1, use_bias=True)
        return tf.squeeze(outputs, [1])

def lrelu(x, leak=0.1):
  return tf.maximum(x, leak * x)
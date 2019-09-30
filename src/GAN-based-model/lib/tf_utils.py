import pickle as pk
import numpy as np
import tensorflow as tf


def build_session(graph: tf.Graph):
    print('Building Session...')
    session = tf.Session(graph=graph)
    with graph.as_default():
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=3)
    return session, saver


def predict_batch(
    sess: tf.Session,
    frame_feat: tf.placeholder,
    frame_len: tf.placeholder,
    frame_temp: tf.placeholder,
    frame_prob: tf.placeholder,
    batch_frame_feat: np.array,
    batch_frame_len: np.array,
):
    feed_dict = {
        frame_feat: batch_frame_feat,
        frame_len: batch_frame_len,
        frame_temp: 0.9
    }
    batch_frame_prob = sess.run(frame_prob, feed_dict=feed_dict)
    return batch_frame_prob

import pickle as pk
import numpy as np
import tensorflow as tf

from evalution import evaluate_frame_result
from lib.data_load import DataLoader


def build_session(graph: tf.Graph):
    print('Building Session...')
    session = tf.Session(graph=graph)
    with graph.as_default():
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=3)
    return session, saver


def output_framewise_prob(
    data_loader: DataLoader,
    sess: tf.Session,
    frame_feat: tf.placeholder,
    frame_len: tf.placeholder,
    frame_temp: tf.placeholder,
    frame_pred: tf.placeholder,
    frame_prob: tf.placeholder,
    output_path: str,
):
    total_frame = 0.0
    total_error = 0.0
    posterior_prob = []
    for batch in data_loader.get_batch(256):
        batch_frame_feat, batch_frame_label, batch_frame_len = \
            batch['source'], batch['frame_label'], batch['source_length']
        feed_dict = {
            frame_feat: batch_frame_feat,
            frame_len: batch_frame_len,
            frame_temp: 0.9,
        }
        [batch_frame_pred, batch_frame_prob] = sess.run(
            [frame_pred, frame_prob],
            feed_dict=feed_dict,
        )
        frame_num, frame_error = evaluate_frame_result(
            batch_frame_pred,
            batch_frame_label,
            batch_frame_len,
            data_loader.phn_mapping,
        )
        total_frame += frame_num
        total_error += frame_error
        posterior_prob.extend(batch_frame_prob)
    total_fer = total_error / total_frame * 100
    print(f'FER: {total_fer:.4f}, {output_path}')
    pk.dump(np.array(posterior_prob), open(output_path, 'wb'))
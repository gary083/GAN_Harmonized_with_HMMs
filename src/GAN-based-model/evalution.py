import os
import numpy as np
import _pickle as pk

from tqdm import tqdm

def frame_eval(sess, g, data_loader):
    total_frame = 0.0
    total_error = 0.0
    for batch_frame_feat, batch_frame_label, batch_frame_len in data_loader.get_batch(256):
        feed_dict = {
            g.frame_feat:  batch_frame_feat,
            g.frame_len:   batch_frame_len,
            g.frame_temp: 0.9
        }
        batch_frame_pred = sess.run(g.frame_pred, feed_dict=feed_dict)
        frame_num, frame_error = evaluate_frame_result(batch_frame_pred, 
                                                       batch_frame_label, 
                                                       batch_frame_len, 
                                                       data_loader.phn_mapping)
        total_frame += frame_num
        total_error += frame_error
    total_fer = total_error / total_frame * 100
    return total_fer

def evaluate_frame_result(frame_pred, frame_label, frame_len, phn_mapping):
    frame_num = np.sum(frame_len)
    frame_error = 0
    for batch_idx in range(len(frame_pred)):
        for idx in range(frame_len[batch_idx]):
            if phn_mapping[frame_pred[batch_idx][idx]] != phn_mapping[frame_label[batch_idx][idx]]:
                frame_error += 1
    return frame_num, frame_error

def output_framewise_prob(sess, g, output_path, data_loader):
    total_frame = 0.0
    total_error = 0.0
    posterior_prob = []
    for batch_frame_feat, batch_frame_label, batch_frame_len in data_loader.get_batch(256):
        feed_dict = {
            g.frame_feat:  batch_frame_feat,
            g.frame_len:   batch_frame_len,
            g.frame_temp: 0.9
        }
        [batch_frame_pred, batch_frame_prob] = sess.run([g.frame_pred, g.frame_prob], feed_dict=feed_dict)
        frame_num, frame_error = evaluate_frame_result(batch_frame_pred, 
                                                       batch_frame_label, 
                                                       batch_frame_len, 
                                                       data_loader.phn_mapping)
        total_frame += frame_num
        total_error += frame_error
        posterior_prob.extend(batch_frame_prob)
    total_fer = total_error / total_frame * 100
    print (f'FER: {total_fer:.4f}, {output_path}')
    pk.dump(np.array(posterior_prob), open(output_path, 'wb'))


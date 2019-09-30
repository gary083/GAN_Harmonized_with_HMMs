import sys

import tensorflow as tf

from evalution import *
from lib.module import *
from models.base import ModelBase
from lib.tf_utils import (
    build_session,
    predict_batch,
)


class SupModel(ModelBase):

    def __init__(self, args, train=True):
        cout_word = 'SUPERVISED MODEL: building    '
        sys.stdout.write(cout_word)
        sys.stdout.flush()
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope('input'):
                self.frame_feat = tf.placeholder(tf.float32, shape=[None, args.feat_max_length, args.feat_dim])
                self.frame_label = tf.placeholder(tf.int32, shape=[None, args.feat_max_length])
                self.frame_len = tf.placeholder(tf.int32, shape=[None])

                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.frame_temp = tf.placeholder(tf.float32, shape=[])

            with tf.variable_scope('generator'):
                self.frame_prob, _, frame_log_prob = frame2phn(self.frame_feat, args, args.sample_temp,
                                                               input_len=self.frame_len)

            if train:
                self.learning_rate = tf.placeholder(tf.float32, shape=[])
                self.seq_loss = sequence_loss(frame_log_prob, self.frame_label, self.frame_len)
                # Optimizer
                variables = [v for v in tf.trainable_variables() if v.name.startswith("generator")]
                train_op = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9)
                gradients = tf.gradients(self.seq_loss, variables)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                self.train_op = train_op.apply_gradients(zip(clipped_gradients, variables))

        self.sess, self.saver = self.build_session()
        sys.stdout.write('\b' * len(cout_word))
        cout_word = 'SUPERVISED MODEL: finish     '
        sys.stdout.write(cout_word + '\n')
        sys.stdout.flush()

    def build_session(self):
        return build_session(self.graph)

    def restore(self, save_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(save_dir))

    def train(
        self,
        args,
        data_loader,
        dev_data_loader,
        **kwargs,
    ):
        print('TRAINING(supervised)...')
        step_seq_loss = 0.0
        max_fer = 100.0

        for epoch in range(1, args.epoch + 1):
            for batch_frame_feat, batch_frame_label, batch_frame_len in data_loader.get_batch(args.batch_size):
                feed_dict = {
                    self.frame_feat: batch_frame_feat,
                    self.frame_label: batch_frame_label,
                    self.frame_len: batch_frame_len,
                    self.learning_rate: args.sup_lr_rate
                }
                run_list = [self.seq_loss, self.train_op]
                seq_loss, _ = self.sess.run(run_list, feed_dict=feed_dict)
                step_seq_loss += seq_loss / data_loader.batch_number

            if epoch % 5 == 0:
                print(f'Epoch: {epoch:5d} seq_loss: {step_seq_loss:.4f}')
                step_fer = frame_eval(self.predict_batch, dev_data_loader)
                print(f'EVAL max: {max_fer:.2f} step: {step_fer:.2f}')
                if step_fer < max_fer: max_fer = step_fer
            step_seq_loss = 0.0
        print('=' * 80)

    def predict_batch(self, batch_frame_feat, batch_frame_len):
        return predict_batch(
            self.sess,
            self.frame_feat,
            self.frame_len,
            self.frame_temp,
            self.frame_prob,
            batch_frame_feat,
            batch_frame_len,
        )

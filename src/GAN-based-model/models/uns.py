import sys

import tensorflow as tf

from evalution import (
    frame_eval
)
from lib.discriminator import weak_discriminator
from lib.module import (
    segment_loss,
    frame2phn,
    generate_inter_sample,
    generate_real_sample,
    creating_embedding_matrix,
    compute_penalty,
)
from models.base import ModelBase
from lib.tf_utils import (
    build_session,
    predict_batch,
)


class UnsModel(ModelBase):

    def __init__(self, config, train=True):
        cout_word = 'UNSUPERVISED MODEL: building    '
        sys.stdout.write(cout_word)
        sys.stdout.flush()
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope('input'):
                # Acoustic Data
                #   Framewise Feature
                self.frame_feat = tf.placeholder(
                    tf.float32,
                    shape=[None, config.feat_max_length, config.feat_dim],
                )
                self.frame_len = tf.placeholder(tf.int32, shape=[None])
                #   Sampled Feature
                self.sample_feat = tf.placeholder(
                    tf.float32,
                    shape=[None, config.phn_max_length, config.feat_dim],
                )
                self.sample_len = tf.placeholder(tf.int32, shape=[None])
                self.sample_rep = tf.placeholder(tf.int32, shape=[None])

                # Real Data
                self.target_idx = tf.placeholder(tf.int32, shape=[None, config.phn_max_length])
                self.target_len = tf.placeholder(tf.int32, shape=[None])

                self.frame_temp = tf.placeholder(tf.float32, shape=[])
                self.global_step = tf.Variable(0, name='global_step', trainable=False)

            with tf.variable_scope('generator'):
                # Get generated phoneme sequence
                self.fake_sample, _, _ = frame2phn(
                    self.sample_feat,
                    config,
                    self.frame_temp,
                    input_len=self.sample_len,
                    reuse=False,
                )

                # Get framewise phoneme distribution
                self.frame_prob, _, _ = frame2phn(
                    self.frame_feat,
                    config,
                    self.frame_temp,
                    input_len=self.frame_len,
                    reuse=True,
                )

            with tf.variable_scope('discriminator'):
                # Get real phoneme sequence
                self.real_sample = generate_real_sample(
                    self.target_idx,
                    self.target_len,
                    config.phn_size,
                )
                inter_sample = generate_inter_sample(self.real_sample, self.fake_sample)

                # weak discriminator
                emb = creating_embedding_matrix(config.phn_size, config.dis_emb_size, 'emb')
                real_sample_pred = weak_discriminator(
                    self.real_sample,
                    emb,
                    config.dis_hidden_1_size,
                    config.dis_hidden_2_size,
                    reuse=False,
                )
                fake_sample_pred = weak_discriminator(
                    self.fake_sample,
                    emb,
                    config.dis_hidden_1_size,
                    config.dis_hidden_2_size,
                    reuse=True,
                )
                inter_sample_pred = weak_discriminator(
                    inter_sample,
                    emb,
                    config.dis_hidden_1_size,
                    config.dis_hidden_2_size,
                    reuse=True,
                )

                # gradient penalty
                gradient_penalty = compute_penalty(inter_sample_pred, inter_sample)

            if train:
                self.learning_rate = tf.placeholder(tf.float32, shape=[])

                with tf.variable_scope('segmental_loss'):
                    sep_size = (config.batch_size * config.repeat) // 2
                    self.seg_loss = segment_loss(
                        self.fake_sample[:sep_size],
                        self.fake_sample[sep_size:],
                        repeat_num=self.sample_rep,
                    )

                with tf.variable_scope('discriminator_loss'):
                    self.real_score = tf.reduce_mean(real_sample_pred)
                    self.fake_score = tf.reduce_mean(fake_sample_pred)
                    self.dis_loss = -(self.real_score - self.fake_score) +\
                        config.penalty_ratio * gradient_penalty

                with tf.variable_scope('generator_loss'):
                    self.gen_loss = -(self.fake_score - self.real_score) +\
                        config.seg_loss_ratio * self.seg_loss

                self.dis_variables = [
                    v for v in tf.trainable_variables() if v.name.startswith("discriminator")
                ]
                self.gen_variables = [
                    v for v in tf.trainable_variables() if v.name.startswith("generator")
                ]

                # Discriminator optimizer
                train_dis_op = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9)
                gradients = tf.gradients(self.dis_loss, self.dis_variables)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                self.train_dis_op = train_dis_op.apply_gradients(
                    zip(clipped_gradients, self.dis_variables)
                )

                # Generator optimizer
                train_gen_op = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9)
                gradients = tf.gradients(self.gen_loss, self.gen_variables)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                self.train_gen_op = train_gen_op.apply_gradients(
                    zip(clipped_gradients, self.gen_variables)
                )

        self.sess, self.saver = self.build_session()
        sys.stdout.write('\b' * len(cout_word))
        cout_word = 'UNSUPERVISED MODEL: finish     '
        sys.stdout.write(cout_word + '\n')
        sys.stdout.flush()

    def build_session(self):
        return build_session(self.graph)

    def train(
        self,
        config,
        data_loader,
        dev_data_loader=None,
        aug=False,
    ):
        print('TRAINING(unsupervised)...')
        if aug:
            get_target_batch = data_loader.get_aug_target_batch
        else:
            get_target_batch = data_loader.get_target_batch

        batch_size = config.batch_size * config.repeat
        step_gen_loss, step_dis_loss, step_seg_loss = 0.0, 0.0, 0.0
        max_fer = 100.0
        frame_temp = 0.9
        for step in range(1, config.step + 1):
            if step == 8000:
                frame_temp = 0.8
            if step == 12000:
                frame_temp = 0.7

            for _ in range(config.dis_iter):
                batch_sample_feat, batch_sample_len, batch_repeat_num, _ = data_loader.get_sample_batch(
                    config.batch_size,
                    repeat=config.repeat,
                )
                batch_target_idx, batch_target_len = get_target_batch(batch_size)

                feed_dict = {
                    self.sample_feat: batch_sample_feat,
                    self.sample_len: batch_sample_len,
                    self.target_idx: batch_target_idx,
                    self.target_len: batch_target_len,
                    self.learning_rate: config.dis_lr,
                    self.frame_temp: frame_temp
                }

                run_list = [self.dis_loss, self.train_dis_op]
                dis_loss, _ = self.sess.run(run_list, feed_dict=feed_dict)

            for _ in range(config.gen_iter):
                batch_sample_feat, batch_sample_len, batch_repeat_num, _ = data_loader.get_sample_batch(
                    config.batch_size,
                    repeat=config.repeat,
                )
                batch_target_idx, batch_target_len = get_target_batch(batch_size)

                feed_dict = {
                    self.sample_feat: batch_sample_feat,
                    self.sample_len: batch_sample_len,
                    self.target_idx: batch_target_idx,
                    self.target_len: batch_target_len,
                    self.sample_rep: batch_repeat_num,
                    self.learning_rate: config.gen_lr,
                    self.frame_temp: frame_temp
                }

                run_list = [self.gen_loss, self.seg_loss, self.train_gen_op, self.fake_sample]
                gen_loss, seg_loss, _, smaple = self.sess.run(run_list, feed_dict=feed_dict)

            step_gen_loss += gen_loss / config.print_step
            step_dis_loss += dis_loss / config.print_step
            step_seg_loss += seg_loss / config.print_step

            if step % config.print_step == 0:
                print(
                    f'Step: {step:5d} '
                    f'dis_loss: {step_gen_loss:.4f} '
                    f'gen_loss: {step_dis_loss:.4f} '
                    f'seg_loss: {step_seg_loss:.4f}'
                )
                step_gen_loss, step_dis_loss, step_seg_loss = 0.0, 0.0, 0.0

            if step % config.eval_step == 0:
                step_fer = frame_eval(self.predict_batch, dev_data_loader)
                print(f'EVAL max: {max_fer:.2f} step: {step_fer:.2f}')
                if step_fer < max_fer:
                    max_fer = step_fer
                    self.saver.save(self.sess, config.save_path)

        print('=' * 80)

    def restore(self, save_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(save_dir))

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

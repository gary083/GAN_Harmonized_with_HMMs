import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
import _pickle as pk
import numpy as np
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter

from src.data.dataLoader import get_data_loader, get_dev_data_loader, sampler
from src.models.gan_wrapper import GenWrapper, DisWrapper
from src.lib.utils import gen_real_sample, pad_sequence
from src.lib.metrics import frame_eval


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class UnsModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        cout_word = 'UNSUPERVISED MODEL: building    '
        sys.stdout.write(cout_word)
        sys.stdout.flush()

        self.config = config
        self.step = 0

        self.gen_model = GenWrapper(config.feat_dim,
                                    config.phn_size,
                                    config.gen_hidden_size).to(device)
        self.dis_model = DisWrapper(config.phn_size,
                                    config.dis_emb_size,
                                    config.dis_hidden_1_size,
                                    config.dis_hidden_2_size,
                                    max_len=config.phn_max_length).to(device)

        self.gen_optim = torch.optim.Adam(self.gen_model.parameters(),
                                          lr=config.gen_lr, betas=(0.5, 0.9))
        self.dis_optim = torch.optim.Adam(self.dis_model.parameters(),
                                          lr=config.dis_lr, betas=(0.5, 0.9))

        sys.stdout.write('\b'*len(cout_word))
        cout_word = 'UNSUPERVISED MODEL: finish     '
        sys.stdout.write(cout_word+'\n')
        sys.stdout.flush()

    def forward(self, sample_feat, sample_len, target_idx, target_len, frame_temp,
                intra_diff_num=None):
        sample_feat = sample_feat.to(device)

        soft_prob = self.gen_model(sample_feat, frame_temp, sample_len)
        fake_sample = soft_prob
        real_sample = gen_real_sample(target_idx, target_len, self.config.phn_size).to(device)

        if intra_diff_num is not None:
            batch_size = soft_prob.size(0) // 2
            intra_diff_num = intra_diff_num.to(device)

            g_loss = self.dis_model.calc_g_loss(real_sample, target_len,
                                                fake_sample, sample_len)
            seg_loss = self.gen_model.calc_intra_loss(soft_prob[:batch_size],
                                                      soft_prob[batch_size:],
                                                      intra_diff_num)
            return g_loss + self.config.seg_loss_ratio*seg_loss, seg_loss, fake_sample

        else:
            d_loss, gp_loss = self.dis_model.calc_d_loss(real_sample, target_len,
                                                         fake_sample, sample_len)
            return d_loss + self.config.penalty_ratio*gp_loss, gp_loss

    def train(self, train_data_set, dev_data_set=None, aug=False):
        print ('TRAINING(unsupervised)...')
        self.log_writer = SummaryWriter(self.config.save_path)

        ######################################################################
        # Build dataloader
        #
        train_source, train_target = get_data_loader(train_data_set,
                                                     batch_size=self.config.batch_size,
                                                     repeat=self.config.repeat)

        train_source = sampler(train_source)
        train_target = sampler(train_target)

        gen_loss, dis_loss, seg_loss, gp_loss = 0, 0, 0, 0
        step_gen_loss, step_dis_loss, step_seg_loss, step_gp_loss = 0, 0, 0, 0
        max_fer = 100.0
        frame_temp = 0.9

        self.gen_model.train()
        self.dis_model.train()

        t = trange(self.config.step)
        for step in t:
            self.step += 1
            if self.step == 8000: fram_temp = 0.8
            if self.step == 12000: fram_temp = 0.7

            for _ in range(self.config.dis_iter):
                self.dis_optim.zero_grad()
                sample_feat, sample_len, intra_diff_num = next(train_source)
                target_idx, target_len = next(train_target)

                dis_loss, gp_loss = self.forward(sample_feat, sample_len,
                                                 target_idx, target_len,
                                                 frame_temp)
                dis_loss.backward()
                d_clip_grad = nn.utils.clip_grad_norm_(self.dis_model.parameters(), 5.0)
                self.dis_optim.step()

                dis_loss = dis_loss.item()
                gp_loss = gp_loss.item()
                t.set_postfix(dis_loss=f'{dis_loss:.2f}',
                              gp_loss=f'{gp_loss:.2f}',
                              gen_loss=f'{gen_loss:.2f}',
                              seg_loss=f'{seg_loss:.5f}')

            self.write_log('D_Loss', {"dis_loss": dis_loss,
                                      "gp_loss": gp_loss})

            for _ in range(self.config.gen_iter):
                self.gen_optim.zero_grad()
                sample_feat, sample_len, intra_diff_num = next(train_source)
                target_idx, target_len = next(train_target)

                gen_loss, seg_loss, fake_sample = self.forward(sample_feat, sample_len,
                                                               target_idx, target_len,
                                                               frame_temp, intra_diff_num)
                gen_loss.backward()
                g_clip_grad = nn.utils.clip_grad_norm_(self.gen_model.parameters(), 5.0)
                self.gen_optim.step()

                gen_loss = gen_loss.item()
                seg_loss = seg_loss.item()
                t.set_postfix(dis_loss=f'{dis_loss:.2f}',
                              gp_loss=f'{gp_loss:.2f}',
                              gen_loss=f'{gen_loss:.2f}',
                              seg_loss=f'{seg_loss:.5f}')

            self.write_log('G_Loss', {"gen_loss": gen_loss,
                                      'seg_loss': seg_loss})

            ######################################################################
            # Update & print losses
            #
            step_gen_loss += gen_loss / self.config.print_step
            step_dis_loss += dis_loss / self.config.print_step
            step_seg_loss += seg_loss / self.config.print_step
            step_gp_loss += gp_loss / self.config.print_step

            if self.step % self.config.print_step == 0:
                tqdm.write(f'Step: {self.step:5d} '+
                           f'dis_loss: {step_dis_loss:.4f} '+
                           f'gp_loss: {step_gp_loss:.4f} '+
                           f'gen_loss: {step_gen_loss:.4f} '+
                           f'seg_loss: {step_seg_loss:.4f}')
                step_gen_loss, step_dis_loss, step_seg_loss, step_gp_loss = 0, 0, 0, 0

            ######################################################################
            # Evaluation
            #
            if self.step % self.config.eval_step == 0:
                step_fer = self.dev(dev_data_set)
                tqdm.write(f'EVAL max: {max_fer:.2f} step: {step_fer:.2f}')
                if step_fer < max_fer: 
                    max_fer = step_fer
                    self.save_ckpt()
                self.gen_model.train()
        print ('='*80)

    def dev(self, dev_data_set):
        dev_source = get_dev_data_loader(dev_data_set, batch_size=256) 
        self.gen_model.eval()
        fers, fnums = 0, 0
        for feat, frame_label, length in dev_source:
            prob = self.gen_model(feat.to(device), mask_len=length).detach().cpu().numpy()
            pred = prob.argmax(-1)
            frame_label = frame_label.numpy()
            frame_error, frame_num, _ = frame_eval(pred, frame_label, length)
            fers += frame_error
            fnums += frame_num
        step_fer = fers / fnums * 100
        return step_fer

    def test(self, dev_data_set, file_path):
        dev_source = get_dev_data_loader(dev_data_set, batch_size=256) 
        self.gen_model.eval()
        fers, fnums = 0, 0
        probs = []
        for feat, frame_label, length in dev_source:
            feat = pad_sequence(feat, max_len=self.config.feat_max_length)
            prob = self.gen_model(feat.to(device), mask_len=length).detach().cpu().numpy()
            pred = prob.argmax(-1)
            frame_label = frame_label.numpy()
            frame_error, frame_num, _ = frame_eval(pred, frame_label, length)

            probs.extend(prob)
            fers += frame_error
            fnums += frame_num
        step_fer = fers / fnums * 100
        print(step_fer)
        print(np.array(probs).shape)
        pk.dump(np.array(probs), open(file_path, 'wb'))

    def save_ckpt(self):
        ckpt_path = os.path.join(self.config.save_path, "ckpt_{}.pth".format(self.step))
        torch.save({
            "gen_model": self.gen_model.state_dict(),
            "dis_model": self.dis_model.state_dict(),
            "gen_optim": self.gen_optim.state_dict(),
            "dis_optim": self.dis_optim.state_dict(),
            "step": self.step
        }, ckpt_path)

    def load_ckpt(self, load_path):
        print('\033[K' + "[INFO]", "Load model from: " + load_path)
        ckpt = torch.load(load_path)
        self.gen_model.load_state_dict(ckpt['gen_model'])
        self.dis_model.load_state_dict(ckpt['dis_model'])
        self.gen_optim.load_state_dict(ckpt['gen_optim'])
        self.dis_optim.load_state_dict(ckpt['dis_optim'])
        self.step = ckpt['step']

    def write_log(self, val_name, val_dict):
        self.log_writer.add_scalars(val_name, val_dict, self.step)


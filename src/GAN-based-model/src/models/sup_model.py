import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
import _pickle as pk
import numpy as np
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter

from src.data.dataLoader import get_dev_data_loader, get_sup_data_loader
from src.models.generator import Frame2Phn
from src.lib.utils import pad_sequence
from src.lib.metrics import frame_eval


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SupModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        cout_word = 'SUPERVISED MODEL: building    '
        sys.stdout.write(cout_word)
        sys.stdout.flush()

        self.config = config
        self.epoch = 0
        self.step = 0

        self.model = Frame2Phn(config.feat_dim, config.phn_size, config.gen_hidden_size).to(device)
        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr=config.gen_lr, betas=(0.5, 0.9))

        sys.stdout.write('\b'*len(cout_word))
        cout_word = 'SUPERVISED MODEL: finish     '
        sys.stdout.write(cout_word+'\n')
        sys.stdout.flush()

    def train(self, train_data_set, dev_data_set=None, aug=False):
        print ('TRAINING(supervised)...')
        self.log_writer = SummaryWriter(self.config.save_path)

        data_loader = get_sup_data_loader(train_data_set, batch_size=self.config.batch_size) 
        step_seq_loss = 0.0
        max_fer = 100.0
        self.model.train()

        t = trange(self.epoch, self.config.epoch)
        for epoch in t:
            self.epoch += 1
            for feat, frame_label, length in tqdm(data_loader):
                self.optim.zero_grad()
                seq_loss, prob = self.model.calc_seq_loss(feat.to(device), frame_label.to(device))
                seq_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optim.step()

                pred = prob.argmax(-1).detach().cpu().numpy()
                frame_label = frame_label.numpy()
                _, _, fer = frame_eval(pred, frame_label, length)
                t.set_postfix(seq_loss=f'{seq_loss:.4f}', FER=f'{fer:.4f}')

                self.step += 1
                self.log_writer.add_scalar('seq_loss', seq_loss, self.step)
                self.log_writer.add_scalar('FER', fer, self.step)
                step_seq_loss += seq_loss / len(data_loader)

            if self.epoch % 5 == 0:
                tqdm.write(f'Epoch: {self.epoch:5d} '+
                           f'seq_loss: {step_seq_loss:.4f}')

                step_fer = self.dev(dev_data_set)
                tqdm.write(f'EVAL max: {max_fer:.2f} step: {step_fer:.2f}')
                if step_fer < max_fer: 
                    max_fer = step_fer
                    self.save_ckpt()
                self.model.train()
            step_seq_loss = 0
        print ('='*80)

    def dev(self, dev_data_set):
        dev_source = get_dev_data_loader(dev_data_set, batch_size=256) 
        self.model.eval()
        fers = 0
        fnums = 0
        for feat, frame_label, length in dev_source:
            prob = self.model(feat.to(device), mask_len=length).detach().cpu().numpy()
            pred = prob.argmax(-1)
            frame_label = frame_label.numpy()
            frame_error, frame_num, _ = frame_eval(pred, frame_label, length)
            fers += frame_error
            fnums += frame_num
        step_fer = fers / fnums * 100
        return step_fer

    def test(self, dev_data_set, file_path):
        dev_source = get_dev_data_loader(dev_data_set, batch_size=256) 
        self.model.eval()
        fers = 0
        fnums = 0
        probs = []
        for feat, frame_label, length in dev_source:
            feat = pad_sequence(feat, max_len=self.config.feat_max_length)
            prob = self.model(feat.to(device), mask_len=length).detach().cpu().numpy()
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
        ckpt_path = os.path.join(self.config.save_path, "ckpt_{}_{}.pth".format(self.epoch, self.step))
        torch.save({
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "epoch": self.epoch,
            "step": self.step
        }, ckpt_path)

    def load_ckpt(self, load_path):
        print('\033[K' + "[INFO]", "Load model from: " + load_path)
        ckpt = torch.load(load_path)
        self.model.load_state_dict(ckpt['model'])
        self.optim.load_state_dict(ckpt['optim'])
        self.epoch = ckpt['epoch']
        self.step = ckpt['step']


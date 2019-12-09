import sys
import pickle
import numpy as np
import random

import torch
from torch.utils.data import Dataset


class PickleDataset(Dataset):
    """
    Input Arguments:
        config
            concat_window   : concat window size
            phn_max_length  : length of phone sequence
            feat_max_length : length of feat sequence
        feat_path           : feature data
        phn_path            : phone data
        orc_bnd_path        : oracle boundaries
        train_bnd_path      : trained boundaries
        target_path         : text data
        data_length         : num of non-matching data
        name                :
        n_steps             : instead of iterating epochs, random n_step batches

    Instance Variables:
        self.phn2idx        : 48 phns -> 48 indices
        self.idx2phn        : 48 indices -> 48 phns
        self.phn_mapping    : 48 indices -> 39 phns
        self.sil_idx        : 
        self.feat_dim       : 39
        self.data_length
        self.source
        self.target
        self.dev
    """
    def __init__(self, config, feat_path, phn_path, orc_bnd_path,
                 train_bnd_path=None, target_path=None, data_length=None,
                 phn_map_path='./phones.60-48-39.map.txt', name='DATA LOADER',
                 random_batch=False, n_steps=None, mode='train'):
        super().__init__()

        cout_word = f'{name}: loading    '
        sys.stdout.write(cout_word)
        sys.stdout.flush()
        self.concat_window   = config.concat_window
        self.phn_max_length  = config.phn_max_length
        self.feat_max_length = config.feat_max_length
        self.sample_var      = config.sample_var
        self.feat_path       = feat_path
        self.phn_path        = phn_path
        self.orc_bnd_path    = orc_bnd_path
        self.train_bnd_path  = train_bnd_path
        self.target_path     = target_path
        self.random_batch    = random_batch
        self.n_steps         = n_steps

        self.read_phn_map(phn_map_path)

        feats     = pickle.load(open(feat_path, 'rb'))
        orc_bnd   = pickle.load(open(orc_bnd_path, 'rb'))
        phn_label = pickle.load(open(phn_path, 'rb'))
        assert (len(feats) == len(orc_bnd) == len(phn_label))
        
        self.data_length = len(feats) if not data_length else data_length
        self.process_feat(feats[:self.data_length])
        self.process_label(orc_bnd[:self.data_length], phn_label[:self.data_length])

        if train_bnd_path:
            self.process_train_bnd(train_bnd_path)

        if target_path:
            self.process_target(target_path)

        self.create_datasets(mode)

        sys.stdout.write('\b' * len(cout_word))
        cout_word = f'{name}: finish     '
        sys.stdout.write(cout_word + '\n')
        sys.stdout.flush()
        print ('='*80)

    def read_phn_map(self, phn_map_path):
        phn_mapping = {}
        with open(phn_map_path, 'r') as f:
            for line in f:
                if line.strip() != "":
                    p60, p48, p39 = line.split()
                    phn_mapping[p48] = p39

        all_phn = list(phn_mapping.keys())
        assert(len(all_phn) == 48)
        self.phn_size = len(all_phn)
        self.phn2idx        = dict(zip(all_phn, range(len(all_phn))))
        self.idx2phn        = dict(zip(range(len(all_phn)), all_phn))
        self.phn_mapping    = dict([(i, phn_mapping[phn]) for i, phn in enumerate(all_phn)])
        self.sil_idx = self.phn2idx['sil']

    def process_feat(self, feats):
        half_window = (self.concat_window-1) // 2
        self.feat_dim = feats[0].shape[-1]
        self.feats = []
        for feat in feats[:self.data_length]:
            _feat_ = np.concatenate([np.tile(feat[0], (half_window, 1)), feat,
                                     np.tile(feat[-1], (half_window, 1))], axis=0)
            feature = torch.tensor([np.reshape(_feat_[l : l+self.concat_window], [-1])
                                    for l in range(len(feat))])[:self.feat_max_length]
            self.feats.append(feature)

    def process_label(self, orc_bnd, phn_label):
        self.frame_labels = []
        for bnd, phn in zip(orc_bnd, phn_label):
            assert len(bnd) == len(phn) + 1
            frame_label = []
            for prev_b, b, p in zip(bnd, bnd[1:], phn):
                frame_label += [self.phn2idx[p]] * (b-prev_b)
            frame_label += [self.phn2idx[phn[-1]]]
            self.frame_labels.append(torch.tensor(frame_label))

    def process_train_bnd(self, train_bnd_path):
        train_bound = pickle.load(open(train_bnd_path, 'rb'))[:self.data_length]
        assert (len(train_bound) == self.data_length)
        self.train_bnd = []
        self.train_bnd_range = []
        self.train_seq_length = []
        for bound in train_bound:
            bound = torch.tensor(bound)
            self.train_bnd.append(bound[:-1][:self.phn_max_length])
            self.train_bnd_range.append((bound[1:] - bound[:-1])[:self.phn_max_length])
            self.train_seq_length.append(min(len(bound)-1, self.phn_max_length))

    def process_target(self, target_path):
        target_data = [line.strip().split() for line in open(target_path, 'r')]
        target_data = [[self.phn2idx[t] for t in target] for target in target_data]
        self.target_data = [torch.tensor(target).int() for target in target_data]

    def create_datasets(self, mode):
        if mode == 'train':
            if self.random_batch and self.n_steps:
                self.source = RandomSourceDataset(self.data_length,
                                                  self.feats,
                                                  self.train_bnd,
                                                  self.train_bnd_range,
                                                  self.train_seq_length,
                                                  self.n_steps)
                self.target = RandomTargetDataset(self.target_data, self.n_steps)
            elif self.n_steps:
                self.source = SourceDataset(self.data_length,
                                            self.feats,
                                            self.train_bnd,
                                            self.train_bnd_range,
                                            self.train_seq_length)
                self.target = TargetDataset(self.target_data)
        self.dev = DevDataset(self.data_length, self.feats, self.frame_labels)

    def print_parameter(self, target=False):
        print ('Data Loader Parameter:')
        print (f'   phoneme number:  {self.phn_size}')
        print (f'   phoneme length:  {self.phn_max_length}')
        print (f'   feature dim:     {self.feat_dim * self.concat_window}')
        print (f'   feature windows: {self.concat_window}')
        print (f'   feature length:  {self.feat_max_length}')
        print (f'   source size:     {self.data_length}')
        if target:
            print (f'   target size:     {len(self.target)}')
        print (f'   feat_path:       {self.feat_path}')
        print (f'   phn_path:        {self.phn_path}')
        print (f'   orc_bnd_path:    {self.orc_bnd_path}')
        print (f'   train_bnd_path:  {self.train_bnd_path}')
        print (f'   target_path:     {self.target_path}')
        print ('='*80)


class TargetDataset(Dataset):
    def __init__(self, target_data):
        self.target_data = target_data

    def __len__(self):
        return len(self.target_data)

    def __getitem__(self, index):
        feat = self.target_data[index]
        return feat, len(feat)


class RandomTargetDataset(TargetDataset):
    """ For random batch. """
    def __init__(self, target_data, n_steps):
        super().__init__(target_data)
        self.n_steps = n_steps

    def __getitem__(self, index):
        index = random.randrange(super().__len__())
        return super().__getitem__(index)


class SourceDataset(Dataset):
    def __init__(self, data_length, feats, train_bnd, train_bnd_range, train_seq_length):
        self.data_length = data_length
        self.feats = feats
        self.train_bnd = train_bnd
        self.train_bnd_range = train_bnd_range
        self.train_seq_length = train_seq_length

    def __len__(self, ):
        return self.data_length

    def __getitem__(self, index):
        feat = self.feats[index]
        train_bnd = self.train_bnd[index]
        train_bnd_range = self.train_bnd_range[index]
        train_seq_length = self.train_seq_length[index]
        return feat, train_bnd, train_bnd_range, train_seq_length


class RandomSourceDataset(SourceDataset):
    """ For random batch. """
    def __init__(self, data_length, feats, train_bnd, train_bnd_range, train_seq_length, n_steps):
        super().__init__(data_length, feats, train_bnd, train_bnd_range, train_seq_length)
        self.n_steps = n_steps

    def __getitem__(self, index):
        index = random.randrange(super().__len__())
        return super().__getitem__(index)


class DevDataset(Dataset):
    def __init__(self, data_length, feats, frame_labels):
        self.data_length = data_length
        self.feats = feats
        self.frame_labels = frame_labels

    def __len__(self, ):
        return self.data_length

    def __getitem__(self, index):
        feat = self.feats[index]
        frame_label = self.frame_labels[index]
        return feat, frame_label

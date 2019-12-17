import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler, BatchSampler
from src.data.dataset import PickleDataset
from functools import partial
import random


def _collate_source_fn(l, repeat=6):
    batch_size  = len(l)
    feats, bnds, bnd_ranges, seq_lengths = zip(*l)

    feats = pad_sequence(feats, batch_first=True, padding_value=0).repeat(repeat * 2, 1, 1)
    bnds = pad_sequence(bnds, batch_first=True, padding_value=0).repeat(repeat * 2, 1).float()
    bnd_ranges = pad_sequence(bnd_ranges, batch_first=True, padding_value=0).repeat(repeat * 2, 1).float()
    seq_lengths = torch.tensor(seq_lengths).repeat(repeat * 2).int()

    random_pick = torch.clamp(torch.randn_like(bnds) * 0.2 + 0.5, 0, 1)
    sample_frame = torch.round(bnds + random_pick * bnd_ranges).long()
    sample_source = feats[torch.arange(batch_size * 2 * repeat).reshape(-1, 1), sample_frame]
    intra_diff_num = (sample_frame[:batch_size * repeat] != sample_frame[batch_size * repeat:]).sum(1).int()
    return sample_source, seq_lengths, intra_diff_num 

def _collate_target_fn(l):
    feats, seq_lengths = zip(*l)

    feats = pad_sequence(feats, batch_first=True, padding_value=0).long()
    seq_lengths = torch.tensor(seq_lengths).int()
    return feats, seq_lengths

def _collate_sup_fn(l, padding_label=-100):
    feats, frame_labels = zip(*l)
    lengths = torch.tensor([len(feat) for feat in feats])

    feats = pad_sequence(feats, batch_first=True, padding_value=0)
    frame_labels = pad_sequence(frame_labels, batch_first=True, padding_value=padding_label)
    return feats, frame_labels, lengths


class RandomBatchSampler(BatchSampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield random.sample(range(len(self.data_source)), self.batch_size)

    def __len__(self):
        return len(self.data_source)


def get_data_loader(dataset, batch_size, repeat=6, random_batch=True, shuffle=False, drop_last=True):
    assert random_batch
    source_collate_fn = partial(_collate_source_fn, repeat=repeat)
    target_collate_fn = _collate_target_fn
    if random_batch:
        source = DataLoader(dataset.source,
                            batch_sampler=RandomBatchSampler(dataset.source, batch_size//2),
                            collate_fn=source_collate_fn)
        target = DataLoader(dataset.target,
                            batch_sampler=RandomBatchSampler(dataset.source, batch_size*6),
                            collate_fn=target_collate_fn)
    return source, target

def get_dev_data_loader(dataset, batch_size, shuffle=False, drop_last=False):
    _collate_dev_fn = partial(_collate_sup_fn, padding_label=-100)
    loader = DataLoader(dataset.dev, batch_size=batch_size, collate_fn=_collate_dev_fn,
                        shuffle=shuffle, drop_last=drop_last)
    return loader

def get_sup_data_loader(dataset, batch_size, shuffle=True, drop_last=True):
    loader = DataLoader(dataset.dev, batch_size=batch_size, collate_fn=_collate_sup_fn,
                        shuffle=shuffle, drop_last=drop_last)
    return loader

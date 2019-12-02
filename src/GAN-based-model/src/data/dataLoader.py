import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler
from torch.distributions.normal import Normal
from src.data.dataset import PickleDataset
from functools import partial
from src.lib.utils import pad_sequence as pad_unsort_sequence

def _collate_source_fn(l, repeat=6):
    l.sort(key=lambda x: x[0].shape[1], reverse=True)
    batch_size  = len(l)
    feats, bnds, bnd_ranges, seq_lengths = zip(*l)

    feats = pad_sequence(feats, batch_first=True, padding_value=0).repeat(repeat * 2, 1, 1)
    bnds = pad_sequence(bnds, batch_first=True, padding_value=0).repeat(repeat * 2, 1).float()
    bnd_ranges = pad_sequence(bnd_ranges, batch_first=True, padding_value=0).repeat(repeat * 2, 1).float()
    seq_lengths = torch.tensor(seq_lengths).repeat(repeat * 2).int()

    random_pick = torch.clamp(torch.randn_like(bnds) * 0.2 + 0.5, 0, 1)
    sample_frame = torch.round(bnds + random_pick * bnd_ranges).long()
    sample_source = feats[torch.arange(batch_size * 2 * repeat).reshape(-1, 1), sample_frame]
    # repeat_num: should be named diff_num, used for calculating loss
    intra_diff_num = (sample_frame[:batch_size * repeat] != sample_frame[batch_size * repeat:]).sum(1).int()
    return sample_source, seq_lengths, intra_diff_num 

def _collate_target_fn(l):
    l.sort(key=lambda x: x[0].shape[0], reverse=True)
    feats, seq_lengths = zip(*l)

    feats = pad_sequence(feats, batch_first=True, padding_value=0).long()
    seq_lengths = torch.tensor(seq_lengths).int()
    return feats, seq_lengths

def _collate_dev_fn(l):
    # l.sort(key=lambda x: x[0].shape[0], reverse=True)
    feats, frame_labels = zip(*l)

    feats, _ = pad_unsort_sequence(feats, pad_value=0)
    frame_labels, lengths = pad_unsort_sequence(frame_labels, pad_value=0, device='cpu')
    return feats, frame_labels, lengths

def _collate_sup_fn(l):
    l.sort(key=lambda x: x[0].shape[0], reverse=True)
    feats, frame_labels = zip(*l)
    # print(max([len(f) for f in feats]), max([len(f) for f in frame_labels]))

    feats, _ = pad_unsort_sequence(feats, pad_value=0)
    frame_labels, lengths = pad_unsort_sequence(frame_labels, pad_value=-100, device='cpu')
    return feats, frame_labels, lengths

def get_data_loader(dataset, batch_size, repeat=6, random_batch=True, shuffle=True, drop_last=True):
    assert random_batch
    source_collate_fn = partial(_collate_source_fn, repeat=repeat)
    source = DataLoader(dataset.source, batch_size=batch_size//2,
                        collate_fn=source_collate_fn, shuffle=shuffle, drop_last=drop_last)
    target_collate_fn = _collate_target_fn
    target = DataLoader(dataset.target, batch_size=batch_size*6,
                        collate_fn=target_collate_fn, shuffle=shuffle, drop_last=drop_last)
    return source, target

def get_dev_data_loader(dataset, batch_size, shuffle=False, drop_last=False):
    loader = DataLoader(dataset.dev, batch_size=batch_size, collate_fn=_collate_dev_fn,
                        shuffle=shuffle, drop_last=drop_last)
    return loader

def get_sup_data_loader(dataset, batch_size, shuffle=True, drop_last=True):
    loader = DataLoader(dataset.dev, batch_size=batch_size, collate_fn=_collate_sup_fn,
                        shuffle=shuffle, drop_last=drop_last)
    return loader

def sampler(data_loader):
    while True:
        for data in data_loader:
            yield data


if __name__ == '__main__':
    """
        args
            concat_window   : concat window size
        phn_max_length      : length of phone sequence
        feat_path           : feature data
        phn_path            : phone data
        orc_bnd_path        : oracle boundaries
        train_bnd_path      : trained boundaries
        target_path         : text data
        sep_number          : num of non-matching data
        name                :
    """
    import argparse
    args = argparse.ArgumentParser()
    audio_path = '/home/r06942045/myData/usr_model/timit_v1/audio'
    bnd_type = 'orc' # ori/orc/...?
    sep_number = None
    timit_set  = 'timit_v1'
    root_path  = '/home/r06942045/myData/usr_model'
    text_path  = f'{root_path}/{timit_set}/text'
    target_path = f'{text_path}/match_lm.48'
    args.concat_window = 11
    args.phn_max_length = 70
    args.feat_max_length = 777
    args.train_feat_path = f'{audio_path}/timit-train-mfcc-nor.pkl'
    args.train_phn_path = f'{audio_path}/timit-train-phn.pkl'
    args.train_orc_bnd_path = f'{audio_path}/timit-train-orc-bnd.pkl'
    args.train_bnd_path = f'{audio_path}/timit-train-{bnd_type}-bnd.pkl'
    args.step = 80000
    args.dev_phn_max_length = 70
    args.dev_feat_path = f'{audio_path}/timit-test-mfcc-nor.pkl'
    args.dev_phn_path = f'{audio_path}/timit-test-phn.pkl'
    args.dev_orc_bnd_path = f'{audio_path}/timit-test-orc-bnd.pkl'
    train_data_set = PickleDataset(args,
                                   args.concat_window,
                                   args.phn_max_length,
                                   args.train_feat_path,
                                   args.train_phn_path,
                                   args.train_orc_bnd_path,
                                   args.train_bnd_path,
                                   target_path,
                                   sep_number=sep_number,
                                   name='DATA LOADER(train)',
                                   random_batch=True,
                                   n_steps=args.step)
    source_loader, target_loader = get_data_loader(train_data_set, 50)
    dev_data_set = PickleDataset(args,
                                 args.concat_window,
                                 args.dev_phn_max_length,
                                 args.dev_feat_path,
                                 args.dev_phn_path,
                                 args.dev_orc_bnd_path,
                                 name='DATA LOADER(dev)',
                                 mode='dev')
    dev_loader = get_dev_data_loader(dev_data_set, 50)
    args.feat_dim = train_data_set.feat_dim * args.concat_window
    args.phn_size = train_data_set.phn_size
    args.sil_idx  = train_data_set.sil_idx
    args.mfcc_dim = train_data_set.feat_dim
    print(args)
    for sample, length in target_loader:
        print(sample.shape)
        print(length)
        print()
        break
    for sample, length, intra_diffN in source_loader:
        print(sample)
        print(sample.shape)
        # print(length.max())
        # print(intra_diffN.dtype)
        print()
        break
    for sample, frame_label, length in dev_loader:
        print(sample.shape)
        print(frame_label)
        print(length)
        print()
        break
    print('='*80)
    generator = sampler(dev_loader)
    print(next(generator)[2])
    for i in range(len(dev_loader)-1):
        print(next(generator)[2])
    print(next(generator)[2])


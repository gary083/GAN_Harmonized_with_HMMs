import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.lib.modules import masked_out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mask_from_lengths(lengths):
    """
    Where should be masked is '0' -> [1, 1, 1, ..., 0, 0]

    Inputs:
        lengths: torch.LongTensor

    Outputs:
        mask: (batch_size, max_len)
    """
    lengths = lengths.to(device)
    max_len = torch.max(lengths).item()
    ids = torch.arange(max_len, device=device)
    mask = (ids < lengths.unsqueeze(1))
    return mask

def gen_real_sample(input_idx, input_len, phn_size):
    return masked_out(F.one_hot(input_idx, phn_size), input_len, axis=1).float()

def pad_seq(sequence, pad_value=0, max_len=float('inf'), device=device):
    batch_size = len(sequence)
    dtype = sequence[0].dtype
    lengths = [min(len(seq), max_len) for seq in sequence]
    if max_len == float('inf'):
        max_len = max(lengths)
    max_len = max(max(lengths), max_len)
    assert min(lengths) > 0, 'Sequence with zero length'
    size = [batch_size, max_len, *(sequence[0].shape[1:])]
    output = torch.ones(size, device=device, dtype=dtype) * pad_value
    for i, seq in enumerate(sequence):
        output[i, :lengths[i]] = seq[:lengths[i]]
    return output, torch.LongTensor(lengths).to(device)

def pad_sequence(sequence, pad_value=0, sample_gram=None, max_len=float('inf'), device=device):
    """ 
    Pad sequence to tensor. If sample_gram is not None, randomly
    sample ngram 30 times for each seq in sequence.
    
    Inputs:
        sequence: list of (len, phn_size)
    """
    if sample_gram is None:
        return pad_seq(sequence, pad_value=pad_value, max_len=max_len, device=device)
    else :
        batch_size = len(sequence)
        phn_size = sequence[0].size(-1)

        n_gram = sample_gram

        sample_sequence = []
        for seq in sequence :
            sample_i = np.random.randint(0, len(seq)-n_gram, 30) # sample 30
            sample_sequence.extend([seq[i: i+n_gram] for i in sample_i])

        output = torch.stack(sample_sequence).to(device)
        lengths = [n_gram] * output.size(0)
    return output, torch.LongTensor(lengths).to(device)


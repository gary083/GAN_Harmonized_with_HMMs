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
    return masked_out(torch.eye(phn_size)[input_idx], input_len)

def pad_sequence(sequence, pad_value=0, sample_gram=None, max_len=None):
    """ 
    Pad sequence to tensor. If sample_gram is not None, randomly
    sample ngram 30 times for each seq in sequence.
    
    Inputs:
        sequence: list of (len, phn_size)
    """
    batch_size = len(sequence)
    phn_size = sequence[0].size(-1)

    if sample_gram is None:
        if max_len is None:
            lengths = [len(seq) for seq in sequence]
            max_len = max(lengths)
        else:
            lengths = [min(len(seq), max_len) for seq in sequence]
        size = [batch_size, max_len, phn_size]

        assert min(lengths) > 0, 'Sequence with zero length'
        output = torch.ones(size, device=device) * pad_value
        for i, seq in enumerate(sequence) :
            output[i, :lengths[i]] = seq[:lengths[i]]
    else :
        n_gram = sample_gram

        sample_sequence = []
        for seq in sequence :
            sample_i = np.random.randint(0, len(seq)-n_gram, 30) # sample 30
            sample_sequence.extend([seq[i: i+n_gram] for i in sample_i])

        output = torch.stack(sample_sequence).to(device)
        lengths = [n_gram] * output.size(0)
    return output, torch.LongTensor(lengths).to(device)


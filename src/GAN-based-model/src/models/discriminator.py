import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.lib.utils import get_mask_from_lengths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResBlock(nn.Module):
    def __init__(self, dim, kernel):
        super().__init__()

        self.res_block = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(dim, dim, kernel, padding=(kernel-1)//2),
            nn.LeakyReLU(),
            nn.Conv1d(dim, dim, kernel, padding=(kernel-1)//2),
        )

    def forward(self, inputs):
        outputs = self.res_block(inputs)
        return inputs + (outputs * 0.3)


class Discriminator(nn.Module):
    """ Not used.
    Arguments:
        dim: channels
    """
    def __init__(self, input_dim, hidden_dim, output_dim, kernel):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1),
            ResBlock(hidden_dim, kernel),
            ResBlock(hidden_dim, kernel),
            ResBlock(hidden_dim, kernel),
            ResBlock(hidden_dim, kernel),
        )
        self.linear = nn.Linear(output_dim, 1)

    def forward(self, inputs, inputs_len=None):
        outputs = self.block(inputs)
        outputs = torch.flatten(outputs, start_dim=1)
        outputs = self.linear(outputs)
        return outputs


class WeakDiscriminator(nn.Module):
    """ Two layers with second layer very few channels
    Arguments:
        dim: channels
    """
    def __init__(self, phn_size, dis_emb_dim, hidden_dim1, hidden_dim2, ngram=None, max_len=None):
        super().__init__()
        self.ngram = ngram
        self.max_len = max_len

        self.emb_bag = nn.Embedding(phn_size, dis_emb_dim)
        self.conv_1 = nn.ModuleList([
            nn.Conv1d(dis_emb_dim, hidden_dim1, 3, padding=1),
            nn.Conv1d(dis_emb_dim, hidden_dim1, 5, padding=2),
            nn.Conv1d(dis_emb_dim, hidden_dim1, 7, padding=3),
            nn.Conv1d(dis_emb_dim, hidden_dim1, 9, padding=4),
        ])
        self.lrelu_1 = nn.LeakyReLU()
        self.conv_2 = nn.ModuleList([
            nn.Conv1d(4*hidden_dim1, hidden_dim2, 3, padding=1),
            nn.Conv1d(4*hidden_dim1, hidden_dim2, 3, padding=1),
            nn.Conv1d(4*hidden_dim1, hidden_dim2, 3, padding=1),
            nn.Conv1d(4*hidden_dim1, hidden_dim2, 3, padding=1),
        ])
        self.flatten = nn.Flatten()
        self.lrelu_2 = nn.LeakyReLU()
        if ngram is not None:
            self.linear = nn.Linear(ngram*4*hidden_dim2, 1)
        elif max_len is not None:
            self.linear = nn.Linear(max_len*4*hidden_dim2, 1)
        else:
            self.linear = nn.Linear(4*hidden_dim2, 1)

    def embedding(self, x):
        return x @ self.emb_bag.weight

    def mask_pool(self, output, lengths=None):
        """ Mean pooling of masked elements """
        if input_len is None:
            return output.mean(1)
        mask = get_mask_from_lengths(lengths).unsqueeze(-1)
        output = output * mask
        return output.sum(1) / mask.sum(1)
    
    def forward(self, inputs, inputs_len=None):
        outputs = self.embedding(inputs)
        outputs = outputs.transpose(1, 2)
        outputs = torch.cat([conv(outputs) for conv in self.conv_1], dim=1)
        outputs = self.lrelu_1(outputs)
        outputs = torch.cat([conv(outputs) for conv in self.conv_2], dim=1)
        outputs = outputs.transpose(1, 2)
        if self.ngram is not None or self.max_len is not None:
            # (B, T, D) -> (B, T*D)
            outputs = self.flatten(outputs)
        else:
            # (B, T, D) -> (B, D)
            output = self.mask_pool(outputs, inputs_len)
        outputs = self.lrelu_2(outputs)
        outputs = self.linear(outputs)
        return outputs

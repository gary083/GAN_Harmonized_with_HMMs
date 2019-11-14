import torch
import torch.nn as nn
import torch.nn.functional as F

from src.lib.modules import MLP, masked_out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Frame2Phn(nn.Module):
    def __init__(self, input_size, output_size, hidden):
        super().__init__()
        if type(hidden) == int:
            hidden = str(hidden)
        elif type(hidden) == list:
            hidden = '_'.join([str(h) for h in hidden])
        self.model = MLP(input_size, output_size, hidden)
        self.softmax = nn.Softmax(-1)

    def forward(self, x, temp=1, mask_len=None):
        """
        Inputs: sampled features
            x: (batch, timesteps, feature_size)
            mask_len: list of int, lengths

        Outputs:
            prob: (batch, timesteps, phn_size)
        """
        prob = self.softmax(self.model(x) / temp)
        if mask_len is not None:
            prob = masked_out(prob, mask_len, axis=1)
        return prob


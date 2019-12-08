import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.discriminator import WeakDiscriminator
from src.models.generator import Frame2Phn
from src.lib.utils import get_mask_from_lengths, pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DisWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = WeakDiscriminator(*args, **kwargs)
        self._spec_init()

    def _spec_init(self):
        for name, para in self.model.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(para.data)
            elif 'bias' in name:
                nn.init.zeros_(para.data)

    def calc_gp(self, real, real_len, fake, fake_len):
        """
        Inputs:
            real, fake: torch.tensor, padded sequence
            real_len, fake_len: torch.LongTensor
        """
        batch_size = min(real.size(0), fake.size(0))
        cut_len = min(real.size(1), fake.size(1))
        real, real_len = real[:batch_size, :cut_len], real_len[:batch_size]
        fake, fake_len = fake[:batch_size, :cut_len], fake_len[:batch_size]

        size = [batch_size] + [1] * (real.dim()-1)
        alpha = torch.rand(size).to(device)

        inter = real + alpha * (fake - real)
        inter_len = torch.stack([real_len, fake_len]).min(0)[0]
        inter_pred = self.model(inter, inter_len)

        gradients = torch.autograd.grad(outputs=inter_pred, inputs=inter,
                grad_outputs=torch.ones(inter_pred.size()).to(device),
                create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def calc_g_loss(self, real, fake):
        """
        Inputs:
            real: list of (len, phn_size)
            fake: list of (len, phn_size)
        """
        real, real_len = pad_sequence(real, sample_gram=self.model.ngram, max_len=self.model.max_len)
        fake, fake_len = pad_sequence(fake, sample_gram=self.model.ngram, max_len=self.model.max_len)

        real_pred = self.model(real, real_len)
        fake_pred = self.model(fake, fake_len)

        g_loss = real_pred.mean() - fake_pred.mean()
        return g_loss

    def calc_d_loss(self, real, fake):
        """
        Inputs:
            real: list of (len, phn_size)
            fake: list of (len, phn_size)
        """
        real, real_len = pad_sequence(real, sample_gram=self.model.ngram, max_len=self.model.max_len)
        fake, fake_len = pad_sequence(fake, sample_gram=self.model.ngram, max_len=self.model.max_len)

        real_pred = self.model(real, real_len)
        fake_pred = self.model(fake, fake_len)

        d_loss = fake_pred.mean() - real_pred.mean()
        gp_loss = self.calc_gp(real, real_len, fake, fake_len)
        return d_loss, gp_loss


class GenWrapper(nn.Module):
    """ Input is already sampled repeatedly (2*batch*repeat, ...). """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = Frame2Phn(*args, **kwargs)
        self._spec_init()

    def _spec_init(self):
        for name, para in self.model.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(para.data)
            elif 'bias' in name:
                nn.init.zeros_(para.data)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # count loss with softmax_logits
    # https://github.com/gary083/GAN_Harmonized_with_HMMs/blob/master/src/GAN-based-model/uns_model.py#L35
    # https://github.com/gary083/GAN_Harmonized_with_HMMs/blob/master/src/GAN-based-model/lib/module.py#L175
    def calc_intra_loss(self, prob1, prob2, intra_diff_num):
        intra = ((prob1 - prob2) ** 2).sum() / intra_diff_num.sum()
        return intra

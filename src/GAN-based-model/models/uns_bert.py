import sys
import os

import torch
from torch import nn
from pytorch_pretrained_bert.modeling import (
    BertPreTrainedModel,
    BertConfig,
    BertEncoder,
)
from pytorch_pretrained_bert.optimization import BertAdam
import numpy as np

from .base import ModelBase
from lib.torch_utils import get_tensor_from_array
from evalution import frame_eval


class UnsBertModel(ModelBase):

    description = "UNSUPERVISED BERT MODEL"

    def __init__(self, config):
        self.config = config
        self.align_layer_idx = -1

        cout_word = f'{self.description}: building    '
        sys.stdout.write(cout_word)
        sys.stdout.flush()

        bert_config = BertConfig(
            vocab_size_or_config_json_file=config.phn_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
        )

        self.bert_model = BertModel(bert_config, config.feat_dim, config.phn_size)
        self.optimizer = BertAdam(
            params=self.bert_model.parameters(),
            lr=3e-5,
            warmup=0.1,
            t_total=config.step,
        )
        if torch.cuda.is_available():
            self.bert_model.cuda()

        sys.stdout.write('\b' * len(cout_word))
        cout_word = f'{self.description}: finish     '
        sys.stdout.write(cout_word + '\n')
        sys.stdout.flush()

    def train(
        self,
        config,
        data_loader,
        dev_data_loader=None,
        aug=False,
    ):
        print('TRAINING(unsupervised)...')
        if aug:
            get_target_batch = data_loader.get_aug_target_batch
        else:
            get_target_batch = data_loader.get_target_batch

        batch_size = config.batch_size * config.repeat
        step_feat_loss, step_target_loss = 0., 0.
        max_fer = 100.0

        for step in range(1, config.step + 1):
            batch_sample_feat, batch_sample_len, batch_repeat_num = data_loader.get_sample_batch(
                config.batch_size,
                repeat=config.repeat,
            )
            self.optimizer.zero_grad()
            feat_loss, _ = self.bert_model.predict_feats(batch_sample_feat, batch_sample_len)
            batch_target_idx, batch_target_len = get_target_batch(batch_size)
            target_loss, _ = self.bert_model.predict_targets(batch_target_idx, batch_target_len)

            total_loss = feat_loss + target_loss
            total_loss.backward()
            self.optimizer.step()

            step_feat_loss += feat_loss.item() / config.print_step
            step_target_loss += target_loss.item() / config.print_step
            if step % config.print_step == 0:
                print(
                    f'Step: {step:5d} '
                    f'feat_loss: {step_feat_loss:.4f} '
                    f'target_loss: {step_target_loss:.4f}'
                )
                step_feat_loss, step_target_loss = 0., 0.

            if step % config.eval_step == 0:
                self.target_embeddings = self.get_target_embeddings()
                step_fer = frame_eval(self.predict_batch, dev_data_loader, batch_size=batch_size)
                print(f'EVAL max: {max_fer:.2f} step: {step_fer:.2f}')
                if step_fer < max_fer:
                    max_fer = step_fer
                    self.save(config.save_path)

        print('=' * 80)

    def save(self, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(
            {
                'state_dict': self.bert_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            },
            os.path.join(save_path, 'checkpoint.pth.tar')
        )

    def restore(self, save_dir):
        checkpoint = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
        self.bert_model = checkpoint['state_dict']
        self.optimizer = checkpoint['optimizer']

    def predict_batch(self, batch_frame_feat, batch_frame_len):
        self.bert_model.eval()
        with torch.no_grad():
            _, encoded_layers = self.bert_model.predict_feats(batch_frame_feat, batch_frame_len)
            predict_embeddings = encoded_layers[self.align_layer_idx]  # shape: (N, T, E)
            predict_inner_products = torch.einsum('nte,pe->ntp', predict_embeddings, self.target_embeddings)
            frame_prob = torch.softmax(predict_inner_products, dim=-1)
            frame_prob = frame_prob.cpu().data.numpy()

        self.bert_model.train()
        return frame_prob

    def get_target_embeddings(self) -> torch.Tensor:
        test_targets = np.arange(self.config.phn_size).reshape(-1, 1)
        test_lens = np.ones(len(test_targets))
        _, encoded_layers = self.bert_model.predict_targets(test_targets, test_lens)
        target_embeddings = encoded_layers[self.align_layer_idx]  # shape (phn_size, E)
        target_embeddings = target_embeddings.squeeze()
        return target_embeddings


class BertModel(BertPreTrainedModel):

    def __init__(
        self,
        config,
        feat_dim,
        phn_size,
        mask_prob=0.15,
        mask_but_no_prob=0.1,
    ):
        super(BertModel, self).__init__(config)
        self.mask_prob = mask_prob
        self.mask_but_no_prob = mask_but_no_prob

        self.feat_embeddings = nn.Linear(feat_dim, config.hidden_size)
        self.feat_mask_vec = nn.Parameter(torch.zeros(feat_dim))

        self.target_embeddings = nn.Embedding(phn_size + 1, config.hidden_size)
        # + 1 for [MASK] token
        self.mask_token = phn_size

        self.encoder = BertEncoder(config)
        self.feat_out_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, feat_dim),
        )
        self.target_out_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, phn_size),
        )
        self.apply(self.init_bert_weights)

    def predict_feats(self, input_feats, seq_lens):
        input_feats = get_tensor_from_array(input_feats)
        attention_mask = self.create_attention_mask(seq_lens, input_feats.shape[1])
        input_mask, predict_mask = self.get_masks(input_feats, self.mask_prob, self.mask_but_no_prob)

        masked_input_feats = input_mask * input_feats + (1 - input_mask) * self.feat_mask_vec
        masked_input_feats *= attention_mask.unsqueeze(2)  # taking care of the paddings

        embedding_output = self.feat_embeddings(masked_input_feats)
        encoded_layers = self.forward(embedding_output, attention_mask)
        output = encoded_layers[-1]
        output = self.feat_out_layer(output)

        to_predict = (1 - predict_mask.squeeze()) * attention_mask  # shape: (N, T)
        mask_l2_loss = torch.sum((output - input_feats) ** 2, dim=2) * to_predict  # shape: (N, T)
        loss = torch.sum(mask_l2_loss) / (torch.sum(to_predict) + 1e-8)
        return loss, encoded_layers

    def predict_targets(self, input_targets, seq_lens):
        input_targets = get_tensor_from_array(input_targets)
        attention_mask = self.create_attention_mask(seq_lens, input_targets.shape[1])

        input_mask, predict_mask = self.get_masks(input_targets, self.mask_prob, self.mask_but_no_prob)
        masked_input_targets = input_targets * input_mask + self.mask_token * (1 - input_mask)
        masked_input_targets *= attention_mask
        masked_input_targets = masked_input_targets.long()

        embedding_output = self.target_embeddings(masked_input_targets)
        encoded_layers = self.forward(embedding_output, attention_mask)
        output = encoded_layers[-1]
        output = self.target_out_layer(output).transpose(1, 2)
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        to_predict = (1 - predict_mask) * attention_mask  # shape: (N, T)
        loss = loss_fn(output, input_targets.long()) * to_predict  # shape: (N, T)
        loss = torch.sum(loss) / (torch.sum(to_predict) + 1e-8)
        return loss, encoded_layers

    @staticmethod
    def create_attention_mask(lens: np.array, max_len: int):
        """
        :param lens: shape (N,)
        convert sequence lengths to sequence masks
        mask: shape:(N, T)
        """
        lens = torch.Tensor(lens).long()
        mask = (torch.arange(max_len).expand(len(lens), max_len) < lens.unsqueeze(1)).float()
        mask = get_tensor_from_array(mask)
        return mask

    @staticmethod
    def get_seq_mask(inp: torch.Tensor, mask_prob: float):
        """
        create mask for mask-lm
        :return: shape: (N, T) or (N, T, 1) according to rank of inp
        0: masked,
        doesn't take care of the padding at the end
        """
        if inp.ndimension() == 3:
            rank2 = inp[:, :, 0]
        else:
            rank2 = inp
        mask = (torch.empty_like(rank2, dtype=torch.float).uniform_() > mask_prob).float()
        if inp.ndimension() == 3:
            mask = mask.unsqueeze(2)
        return mask

    @classmethod
    def get_masks(cls, inp: torch.Tensor, mask_prob, mask_but_no_prob):
        predict_mask = cls.get_seq_mask(inp, mask_prob)
        temp_mask = cls.get_seq_mask(inp, mask_but_no_prob)
        input_mask = 1 - (1 - predict_mask) * temp_mask
        return input_mask, predict_mask

    def forward(self, embedding_output, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(embedding_output[:, :, 0])

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=True,
        )
        return encoded_layers

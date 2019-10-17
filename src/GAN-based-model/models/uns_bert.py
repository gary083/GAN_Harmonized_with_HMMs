import sys
import os
import copy

import torch
from torch import nn
from pytorch_pretrained_bert.modeling import (
    BertPreTrainedModel,
    BertConfig,
    BertLayer,
)
from pytorch_pretrained_bert.optimization import BertAdam
import numpy as np

from .base import ModelBase
from lib.torch_utils import (
    get_tensor_from_array,
    masked_reduce_mean,
    l2_loss,
    cpc_loss,
    intra_segment_loss,
    inter_segment_loss,
)
from evalution import phn_eval


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

        self.bert_model = BertModel(
            bert_config,
            config.feat_dim,
            config.phn_size,
            (config.batch_size * config.repeat) // 2,
        )
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
        step_feat_loss, step_intra_segment_loss, step_inter_segment_loss, step_target_loss = 0., 0., 0., 0.
        max_err = 100.0

        for step in range(1, config.step + 1):
            batch_sample_feat, batch_sample_len, batch_repeat_num, batch_phn_label = data_loader.get_sample_batch(
                config.batch_size,
                repeat=config.repeat,
            )
            self.optimizer.zero_grad()
            feat_loss, intra_s_loss, inter_s_loss = self.bert_model.predict_feats(batch_sample_feat, batch_sample_len, batch_repeat_num)
            batch_target_idx, batch_target_len = get_target_batch(batch_size)
            target_loss = self.bert_model.predict_targets(batch_target_idx, batch_target_len)

            total_loss = 2 * feat_loss + target_loss + 2 * intra_s_loss + inter_s_loss
            total_loss.backward()
            self.optimizer.step()

            step_feat_loss += feat_loss.item() / config.print_step
            step_intra_segment_loss += intra_s_loss.item() / config.print_step
            step_inter_segment_loss += inter_s_loss.item() / config.print_step
            step_target_loss += target_loss.item() / config.print_step
            if step % config.print_step == 0:
                print(
                    f'Step: {step:5d} '
                    f'feat_loss: {step_feat_loss:.2f} '
                    f'intra_segment_loss: {step_intra_segment_loss:.2f} '
                    f'inter_segment_loss: {step_inter_segment_loss:.2f} '
                    f'target_loss: {step_target_loss:.2f}'
                )
                step_feat_loss, step_intra_segment_loss, step_inter_segment_loss, step_target_loss = 0., 0., 0., 0.

            if step % config.eval_step == 0:
                step_err, labels, preds = self.phn_eval(
                    data_loader,
                    batch_size=batch_size,
                    repeat=config.repeat,
                )
                print(f'EVAL max: {max_err:.2f} step: {step_err:.2f}')
                print(f'LABEL  : {" ".join(["%3s" % str(l) for l in labels[0]])}')
                print(f'PREDICT: {" ".join(["%3s" % str(p) for p in preds[0]])}')
                if step_err < max_err:
                    max_err = step_err
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
            predict_target_logits = self.bert_model.predict_targets_from_feats(
                batch_frame_feat,
                batch_frame_len,
            )
            frame_prob = torch.softmax(predict_target_logits, dim=-1)
            frame_prob = frame_prob.cpu().data.numpy()
        self.bert_model.train()
        return frame_prob

    def phn_eval(self, data_loader, batch_size, repeat):
        # self.bert_model.eval()
        with torch.no_grad():
            error_counts = []
            lens = []
            for _ in range(10):
                batch_feat, batch_feat_len, _, batch_phn_label = data_loader.get_sample_batch(
                    batch_size,
                    repeat=repeat,
                )
                batch_target_idx, batch_target_len = data_loader.get_aug_target_batch(batch_size)
                batch_prob = self.bert_model.predict_targets_from_feats(
                    batch_feat, batch_feat_len,
                    batch_target_idx, batch_target_len,
                )
                batch_prob = batch_prob.cpu().data.numpy()
                batch_pred = np.argmax(batch_prob, axis=-1)
                error_count, label_length, sample_labels, sample_preds = phn_eval(batch_pred, batch_feat_len, batch_phn_label, data_loader.phn_mapping)
                error_counts.append(error_count)
                lens.append(label_length)

        # self.bert_model.train()
        err = np.sum(error_counts) / np.sum(lens) * 100
        return err, sample_labels, sample_preds


class BertModel(BertPreTrainedModel):

    def __init__(
        self,
        config,
        feat_dim,
        phn_size,
        sep_size,
        mask_prob=0.15,
        mask_but_no_prob=0.1,
        translate_layer_idx=-1,
        update_ratio=0.99,
    ):
        super(BertModel, self).__init__(config)
        self.mask_prob = mask_prob
        self.mask_but_no_prob = mask_but_no_prob
        self.translate_layer_idx = translate_layer_idx
        self.update_ratio = update_ratio
        self.sep_size = sep_size

        self.feat_embeddings = nn.Linear(feat_dim, config.hidden_size)
        self.feat_mask_vec = nn.Parameter(torch.zeros(feat_dim))

        self.target_embeddings = nn.Embedding(phn_size + 1, config.hidden_size)
        # + 1 for [MASK] token
        self.mask_token = phn_size

        self.translation_dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.shadow_feat_inner = torch.zeros(config.hidden_size)
        self.shadow_target_inner = torch.zeros(config.hidden_size)
        if torch.cuda.is_available():
            self.shadow_feat_inner = self.shadow_feat_inner.cuda()
            self.shadow_target_inner = self.shadow_target_inner.cuda()

        layer = BertLayer(config)
        self.encoder = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        self.dropout = nn.Dropout(p=0.5)
        self.feat_out_layer = nn.Linear(config.hidden_size, feat_dim)
        self.target_out_layer = nn.Linear(config.hidden_size, phn_size)
        self.apply(self.init_bert_weights)

    def predict_feats(self, input_feats, seq_lens, repeats):
        input_feats = get_tensor_from_array(input_feats)
        repeats = get_tensor_from_array(repeats)
        attention_mask = self.create_attention_mask(seq_lens, input_feats.shape[1])
        input_mask, predict_mask = self.get_masks(input_feats, self.mask_prob, self.mask_but_no_prob)

        masked_input_feats = input_mask * input_feats + (1 - input_mask) * self.feat_mask_vec
        masked_input_feats *= attention_mask.unsqueeze(2)  # taking care of the paddings

        embedding_output = self.feat_embeddings(masked_input_feats)
        feat_inner = self.forward(embedding_output, attention_mask, end_layer_idx=self.translate_layer_idx)
        output = self.forward(feat_inner, attention_mask, start_layer_idx=self.translate_layer_idx)
        output = self.feat_out_layer(output)

        to_predict = (1 - predict_mask.squeeze()) * attention_mask  # shape: (N, T)
        loss = cpc_loss(output, input_feats, to_predict, attention_mask)
        # loss = loss + l2_loss(output, input_feats, attention_mask)

        translation_inner = self.translate(feat_inner)
        translated_logits = self.forward(translation_inner, attention_mask, start_layer_idx=self.translate_layer_idx)
        translated_logits = self.target_out_layer(translated_logits)

        intra_s_loss = intra_segment_loss(translated_logits, repeats, attention_mask, self.sep_size)
        inter_s_loss = inter_segment_loss(translated_logits, attention_mask)

        self.update_shadow_variable(self.shadow_feat_inner, feat_inner, attention_mask)
        return loss, intra_s_loss, inter_s_loss

    def predict_targets(self, input_targets, seq_lens):
        input_targets = get_tensor_from_array(input_targets)
        attention_mask = self.create_attention_mask(seq_lens, input_targets.shape[1])

        input_mask, predict_mask = self.get_masks(input_targets, self.mask_prob, self.mask_but_no_prob)
        masked_input_targets = input_targets * input_mask + self.mask_token * (1 - input_mask)
        masked_input_targets *= attention_mask
        masked_input_targets = masked_input_targets.long()

        embedding_output = self.target_embeddings(masked_input_targets)
        target_inner = self.forward(embedding_output, attention_mask, end_layer_idx=self.translate_layer_idx)
        output = self.forward(target_inner, attention_mask, start_layer_idx=self.translate_layer_idx)
        output = self.target_out_layer(output).transpose(1, 2)
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        to_predict = (1 - predict_mask) * attention_mask  # shape: (N, T)
        loss = loss_fn(output, input_targets.long())  # shape: (N, T)
        loss = torch.mean(masked_reduce_mean(loss, to_predict))

        self.update_shadow_variable(self.shadow_target_inner, target_inner, attention_mask)
        return loss

    def predict_targets_from_feats(
            self,
            feats, feat_lens,
            target_idx, target_lens,
    ):
        feats = get_tensor_from_array(feats)
        feat_attention_mask = self.create_attention_mask(feat_lens, feats.shape[1])
        feat_embedding = self.feat_embeddings(feats)
        feat_inner = self.forward(feat_embedding, feat_attention_mask, end_layer_idx=self.translate_layer_idx)
        self.update_shadow_variable(self.shadow_feat_inner, feat_inner, feat_attention_mask)

        target_idx = get_tensor_from_array(target_idx).long()
        target_attention_mask = self.create_attention_mask(target_lens, target_idx.shape[1])
        target_embedding = self.target_embeddings(target_idx)
        target_inner = self.forward(target_embedding, target_attention_mask, end_layer_idx=self.translate_layer_idx)
        self.update_shadow_variable(self.shadow_target_inner, target_inner, target_attention_mask)

        translated_feat_inner = self.translate(feat_inner)
        output = self.forward(translated_feat_inner, feat_attention_mask, start_layer_idx=self.translate_layer_idx)
        output = self.target_out_layer(output)
        return output

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
        predict_mask = cls.get_seq_mask(inp, mask_prob)  # mask_prob of 0s
        temp_mask = cls.get_seq_mask(inp, mask_but_no_prob)
        input_mask = 1 - (1 - predict_mask) * temp_mask  # fewer 0s
        return input_mask, predict_mask

    def forward(self, embedding_output, attention_mask, start_layer_idx=0, end_layer_idx=None):
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

        x = embedding_output
        for layer in self.encoder[start_layer_idx:end_layer_idx]:
            x = layer(x, extended_attention_mask)
            x = self.dropout(x)
        return x

    def update_shadow_variable(self, shadow_var, inner, mask):
        mean_inner = torch.mean(
            masked_reduce_mean(inner, mask),
            dim=0,
        )
        shadow_var *= self.update_ratio
        shadow_var += (1 - self.update_ratio) * mean_inner

    def translate(self, feat_inner):
        translation_vec = (self.shadow_target_inner - self.shadow_feat_inner).detach()
        feat_inner = feat_inner + translation_vec
        feat_inner = self.translation_dense(feat_inner)
        return feat_inner

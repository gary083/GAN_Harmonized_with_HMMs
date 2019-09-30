from unittest import TestCase

import numpy as np
import torch

from ..uns_bert import BertModel


class TestBertUtils(TestCase):

    def setUp(self):
        self.mask_prob = 0.15
        self.mask_but_no_prob = 0.1
        self.N = 10000
        self.T = 100
        self.V = 34
        self.rank_2_inp = torch.ones([self.N, self.T])
        self.rank_3_inp = torch.ones([self.N, self.T, self.V])

    def test_create_attention_mask(self):
        seq_lens = np.array([3, 5, 1, 0, 9])
        max_len = 9
        mask = BertModel.create_attention_mask(seq_lens, max_len)
        mask = mask.data.numpy()
        ans = np.array([
            [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ])
        self.assertTrue(np.all(mask == ans))

    def test_get_seq_mask(self):
        rank2_mask = BertModel.get_seq_mask(self.rank_2_inp, mask_prob=self.mask_prob)
        self.assertAlmostEqual(1 - self.mask_prob, torch.mean(rank2_mask).item(), places=2)
        self.assertTupleEqual(rank2_mask.shape, (self.N, self.T))

        rank3_mask = BertModel.get_seq_mask(self.rank_3_inp, mask_prob=self.mask_prob)
        self.assertAlmostEqual(1 - self.mask_prob, torch.mean(rank3_mask).item(), places=2)
        self.assertTupleEqual(rank3_mask.shape, (self.N, self.T, 1))

    def test_get_masks(self):
        input_mask, predict_mask = BertModel.get_masks(
            self.rank_2_inp,
            self.mask_prob,
            self.mask_but_no_prob,
        )
        self.assertTupleEqual(input_mask.shape, (self.N, self.T))
        self.assertTupleEqual(predict_mask.shape, (self.N, self.T))
        self.assertAlmostEqual(1 - self.mask_prob, torch.mean(predict_mask).item(), places=2)
        self.assertAlmostEqual(
            1 - self.mask_prob * (1 - self. mask_but_no_prob),
            torch.mean(input_mask).item(),
            places=2,
        )

        input_mask, predict_mask = BertModel.get_masks(
            self.rank_3_inp,
            self.mask_prob,
            self.mask_but_no_prob,
        )
        self.assertTupleEqual(input_mask.shape, (self.N, self.T, 1))
        self.assertTupleEqual(predict_mask.shape, (self.N, self.T, 1))
        self.assertAlmostEqual(1 - self.mask_prob, torch.mean(predict_mask).item(), places=2)
        self.assertAlmostEqual(
            1 - self.mask_prob * (1 - self. mask_but_no_prob),
            torch.mean(input_mask).item(),
            places=2,
        )

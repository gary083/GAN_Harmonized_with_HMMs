import _pickle as pk
import sys

import numpy as np
import yaml


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_config(path):
    return AttrDict(yaml.load(open(path, 'r'), Loader=yaml.FullLoader))


class DataLoader:

    def __init__(
        self,
        config,
        feat_path,
        phn_path,
        orc_bnd_path,
        meta_path,
        train_bnd_path=None,
        target_path=None,
        data_length=None,
        phn_map_path='./phones.60-48-39.map.txt',
        name='DATA LOADER',
    ):

        cout_word = f'{name}: loading    '
        sys.stdout.write(cout_word)
        sys.stdout.flush()

        self.phn_max_length = config.phn_max_length
        self.feat_max_length = config.feat_max_length
        self.concat_window = config.concat_window
        self.sample_var = config.sample_var
        self.feat_path = feat_path
        self.phn_path = phn_path
        self.orc_bnd_path = orc_bnd_path
        self.train_bnd_path = train_bnd_path
        self.target_path = target_path

        self.read_phn_map(phn_map_path)

        feat = self.load_pickle(feat_path)
        phn = self.load_pickle(phn_path)
        orc_bnd = self.load_pickle(orc_bnd_path)
        meta = self.load_pickle(meta_path)['prefix']
        assert (len(feat) == len(phn) == len(orc_bnd))

        self.data_length = len(feat) if data_length is None else data_length
        self.process_feat(feat[:self.data_length])
        self.process_label(orc_bnd[:self.data_length], phn[:self.data_length], meta[:self.data_length])

        if train_bnd_path is not None:
            self.process_train_bnd(train_bnd_path)

        if target_path is not None:
            self.process_target(target_path)

        sys.stdout.write('\b' * len(cout_word))
        cout_word = f'{name}: finish     '
        sys.stdout.write(cout_word + '\n')
        sys.stdout.flush()
        print('=' * 80)

    @staticmethod
    def load_pickle(file_name):
        return pk.load(open(file_name, 'rb'))

    def read_phn_map(self, path):
        all_lines = open(path, 'r').read().splitlines()
        phn_mapping = {}
        for line in all_lines:
            if line.strip() == "":
                continue
            phn_mapping[line.split()[1]] = line.split()[2]

        all_phn = list(phn_mapping.keys())
        assert (len(all_phn) == 48)
        self.phn2idx = dict(zip(all_phn, range(len(all_phn))))
        self.idx2phn = dict(zip(range(len(all_phn)), all_phn))
        self.phn_size = len(all_phn)
        self.phn_mapping = {}
        self.sil_idx = self.phn2idx['sil']
        for phn in all_phn:
            self.phn_mapping[self.phn2idx[phn]] = phn_mapping[phn]

    def pad_value(self, seq, value, max_length):
        clip_seq = seq[:max_length]
        pad_lens = [(0, max_length - len(clip_seq))]
        pad_lens.extend([(0, 0) for _ in range(len(seq.shape) - 1)])
        return np.lib.pad(clip_seq, pad_lens,
                          'constant', constant_values=(0, value))

    def process_train_bnd(self, train_bnd_path):
        train_bound = self.load_pickle(train_bnd_path)[:self.data_length]
        assert (len(train_bound) == self.data_length)
        self.train_bnd = np.zeros(shape=[self.data_length, self.phn_max_length], dtype='int32')
        self.train_bnd_range = np.zeros(shape=[self.data_length, self.phn_max_length], dtype='int32')
        self.train_seq_length = np.zeros(shape=[self.data_length], dtype='int32')

        for idx, bnd in enumerate(train_bound):
            self.train_bnd[idx] = self.pad_value(np.array(bnd[:-1]), 0, self.phn_max_length)
            self.train_bnd_range[idx] = self.pad_value(np.array(bnd[1:]) - np.array(bnd[:-1]), 0, self.phn_max_length)
            self.train_seq_length[idx] = len(bnd) - 1

    def process_feat(self, feature):
        """
        :param feature: mfcc / fbank
        """
        self.feat_dim = feature[0].shape[-1]
        self.source_data = np.zeros(shape=[self.data_length, self.feat_max_length, self.feat_dim * self.concat_window],
                                    dtype='float32')
        self.source_data_length = np.zeros(shape=[self.data_length], dtype='int32')

        for idx, feat in enumerate(feature):
            for l in range(len(feat)):
                half_window = int((self.concat_window - 1) / 2)
                if l < half_window:
                    pad_feat = np.tile(feat[0], (half_window - l, 1))
                    concat_feat = np.concatenate([pad_feat, feat[0:l + half_window + 1]], axis=0)
                elif l > len(feat) - half_window - 1:
                    pad_feat = np.tile(feat[-1], (half_window - (len(feat) - l - 1), 1))
                    concat_feat = np.concatenate([feat[l - half_window:len(feat)], pad_feat], axis=0)
                else:
                    concat_feat = feat[l - half_window:l + half_window + 1]

                self.source_data[idx][l] = np.reshape(concat_feat, [-1])
            self.source_data_length[idx] = len(feat)

    def process_label(self, oracle_bound, phoneme, meta):
        """
        label: boundaries, phonemes
        """
        self.frame_label = np.zeros(shape=[self.data_length, self.feat_max_length], dtype='int32')
        self.orc_bnd = np.array(oracle_bound)
        self.sample_phn_label = np.asarray([
            [self.phn2idx[p] for p in phn_seq]
            for phn_seq in phoneme
        ])
        # parse prefix-string to labels, description below
        # https://catalog.ldc.upenn.edu/docs/LDC93S1/timit.readme.html?fbclid=IwAR3DEnjodNL10CaOQCMtQHWovY3I5Hh9JaMYpoHYW-Bz0r6_fXTowH6fjw8
        prefixes = meta
        split_prefixes = [p.split('_') for p in prefixes]
        self.dialect_label, self.int2dialect, self.dialect2int = self.build_label_mapping([p[0] for p in split_prefixes])
        self.sex_label, self.int2sex, self.sex2int = self.build_label_mapping([p[1][0] for p in split_prefixes])
        self.speaker_label, self.int2speaker, self.speaker2int = self.build_label_mapping([p[1][1:] for p in split_prefixes])
        self.text_type_label, self.int2text_type, self.text_type2int = self.build_label_mapping([p[2][:2] for p in split_prefixes])
        self.sentence_label , self.int2sentence_number, self.sentence_number2int = \
            self.build_label_mapping([p[2][2:] for p in split_prefixes])

        for idx, bnd, phn in zip(range(self.data_length), oracle_bound, phoneme):
            assert (len(bnd) == len(phn) + 1)
            prev_b = 0
            for b, p in zip(bnd[1:], phn):
                self.frame_label[idx][prev_b:b] = np.array([self.phn2idx[p]] * (b - prev_b))
                prev_b = b
            self.frame_label[idx][b] = self.phn2idx[p]

    def build_label_mapping(self, original_label: list):
        """
        :param original_label: list of original labels, len=self.data_length
        :return: int_labels: np.array, int2label_map: dict, label2int_map: dict
        """
        label_set = {label for label in original_label}
        label_list = sorted(label_set)
        int2label_map = {i: l for i, l in enumerate(label_list)}
        label2int_map = {l: i for i, l in enumerate(label_list)}
        int_labels = np.array([label2int_map[label] for label in original_label], dtype='int32')
        return int_labels, int2label_map, label2int_map

    def process_target(self, target_path):
        target_data = [line.strip().split() for line in open(target_path, 'r')]
        self.target_data_length = len(target_data)
        self.target_data = np.zeros(shape=[self.target_data_length, self.phn_max_length], dtype='int32')
        self.target_length = np.zeros(shape=[self.target_data_length], dtype='int32')

        for idx, target in enumerate(target_data):
            self.target_data[idx][:len(target)] = np.array([self.phn2idx[t] for t in target])
            self.target_length[idx] = len(target)

    def print_parameter(self, target=False):
        print('Data Loader Parameter:')
        print(f'   phoneme number:  {self.phn_size}')
        print(f'   phoneme length:  {self.phn_max_length}')
        print(f'   feature dim:     {self.feat_dim * self.concat_window}')
        print(f'   feature windows: {self.concat_window}')
        print(f'   feature length:  {self.feat_max_length}')
        print(f'   source size:     {self.data_length}')
        if target:
            print(f'   target size:     {self.target_data_length}')
        print(f'   feat_path:       {self.feat_path}')
        print(f'   phn_path:        {self.phn_path}')
        print(f'   orc_bnd_path:    {self.orc_bnd_path}')
        print(f'   train_bnd_path:  {self.train_bnd_path}')
        print(f'   target_path:     {self.target_path}')
        print('=' * 80)

    def get_sample_batch(self, batch_size, repeat=1):
        batch_size = batch_size // 2
        batch_idx = np.random.choice(self.data_length, batch_size, replace=False)
        batch_idx = np.tile(batch_idx, repeat)
        random_pick = np.clip(
            np.random.normal(
                0.5,
                0.2,
                [batch_size * 2 * repeat, self.phn_max_length]
            ),
            0.0,
            1.0,
        )
        sample_frame = np.around(
            np.tile(self.train_bnd[batch_idx], (2, 1)) +
            random_pick * np.tile(
                self.train_bnd_range[batch_idx], (2, 1)
            )
        ).astype('int32')
        sample_source = np.tile(self.source_data[batch_idx], (2, 1, 1))[
            np.arange(batch_size * 2 * repeat).reshape([-1, 1]), sample_frame
        ]
        repeat_num = np.sum(
            np.not_equal(
                sample_frame[:batch_size * repeat],
                sample_frame[batch_size * repeat:],
            ).astype(np.int32),
            axis=1,
        )
        sample_phn_label = self.sample_phn_label[batch_idx]
        return (
            sample_source,
            np.tile(self.train_seq_length[batch_idx], 2),
            repeat_num,
            np.tile(sample_phn_label, 2),
        )

    def get_target_batch(self, batch_size):
        batch_idx = np.random.choice(self.target_data_length, batch_size, replace=False)
        return self.target_data[batch_idx], self.target_length[batch_idx]

    def get_aug_target_batch(self, batch_size):
        batch_idx = np.random.choice(self.target_data_length, batch_size, replace=False)
        batch_target_data = np.zeros(shape=[batch_size, self.phn_max_length], dtype='int32')
        batch_target_length = np.zeros(shape=[batch_size], dtype='int32')
        for i, (seq, length) in enumerate(zip(self.target_data[batch_idx], self.target_length[batch_idx])):
            new_seq, new_length = self.data_augmentation(seq, length)
            if new_length > self.phn_max_length:
                new_length = self.phn_max_length
            batch_target_data[i][:new_length] = new_seq[:new_length]
            batch_target_length[i] = new_length
        return batch_target_data, batch_target_length

    def data_augmentation(self, seq, length):
        new_seq = []
        for s in seq[:length]:
            if s == self.sil_idx:
                # new_seq.extend([s]*np.random.choice([0, 1, 2], p=[0.04, 0.8, 0.16]))
                new_seq.extend([s])
            else:
                new_seq.extend([s] * np.random.choice([0, 1, 2, 3], p=[0.04, 0.78, 0.17, 0.01]))
        return np.array(new_seq), len(new_seq)

    def generate_batch_number(self, batch_size):
        self.batch_number = (self.data_length - 1) // batch_size + 1

    def reset_batch_pointer(self):
        self.pointer = 0

    def update_pointer(self, batch_size):
        self.pointer += batch_size

    def get_batch(self, batch_size):
        self.generate_batch_number(batch_size)
        self.reset_batch_pointer()

        for i in range(self.batch_number):
            batch_source = self.source_data[self.pointer:self.pointer + batch_size]  # framewise feature
            batch_frame_label = self.frame_label[self.pointer:self.pointer + batch_size]  # framewise phoneme label
            batch_source_length = self.source_data_length[self.pointer:self.pointer + batch_size]  # sequence length
            batch_dialect_label = self.dialect_label[self.pointer:self.pointer + batch_size]
            batch_sex_label = self.sex_label[self.pointer:self.pointer + batch_size]
            batch_speaker_label = self.speaker_label[self.pointer:self.pointer + batch_size]
            batch_text_type_label = self.text_type_label[self.pointer:self.pointer + batch_size]
            batch_sentence_label = self.sentence_label[self.pointer:self.pointer + batch_size]
            self.update_pointer(batch_size)
            yield {
                'source': batch_source,
                'frame_label': batch_frame_label,
                'source_length': batch_source_length,
                'dialect_label': batch_dialect_label,
                'sex_label': batch_sex_label,
                'speaker_label': batch_speaker_label,
                'text_type_label': batch_text_type_label,
                'sentence_label': batch_sentence_label,
            }

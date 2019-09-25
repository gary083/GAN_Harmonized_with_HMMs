import _pickle as cPickle
import os
import sys

import numpy as np
from spectrogram_extractor import *

# mfcc paras
LENGTH = 400 / 2  # offset
# LENGTH = 0
SHIFT = 160


def mfcc_fbank(path):
    with open(path, 'r') as f:
        seq = []
        for i, LINE in enumerate(f.readlines()):
            line = LINE.rstrip()
            if "[" in line:
                continue
            elif "]" in line:
                line = line.split()[:-1]
            else:
                line = line.split()
            for num in line:
                num = float(num)
            seq.append(line)
        return np.array(seq, dtype=np.float32)


def txt(path):
    with open(path, 'r') as f:
        t = f.readlines()[0].rstrip()
        t = t.split()[2:]
        text = ' '.join(t)
        return text


def phn(path, length):
    frame_bounds = []
    with open(path, 'r') as f:
        for i, LINE in enumerate(f.readlines()):
            line = LINE.rstrip()
            start, end, phoneme = line.split()
            start = round((int(start) - LENGTH) / SHIFT)
            end = round((int(end) - LENGTH) / SHIFT)
            frame_bounds.append([phoneme, start, end])
        if frame_bounds[0][1] < 0:
            frame_bounds[0][1] = 0
        if frame_bounds[-1][2] > length - 1:
            frame_bounds[-1][2] = length - 1
    # if frame_bounds[-1][1] >= frame_bounds[-1][2]:
    #     print(f"Error: {path}")
    #     print(frame_bounds[-1], length)
    return frame_bounds


def wrd(path, length):
    frame_bounds = []
    with open(path, 'r') as f:
        for i, LINE in enumerate(f.readlines()):
            line = LINE.rstrip()
            start, end, phoneme = line.split()
            start = round((int(start) - LENGTH) / SHIFT)
            end = round((int(end) - LENGTH) / SHIFT)
            frame_bounds.append([phoneme, start, end])
        if frame_bounds[0][1] < 0:
            frame_bounds[0][1] = 0
        if frame_bounds[-1][2] > length - 1:
            frame_bounds[-1][2] = length - 1
    # if frame_bounds[-1][1] >= frame_bounds[-1][2]:
    #     print(f"Error: {path}")
    #     print(frame_bounds[-1], length)
    return frame_bounds


if __name__ == "__main__":
    root = sys.argv[1]
    timit = sys.argv[2]
    mode = sys.argv[3]  # train or test

    meta_data = {}
    mfcc_feats = []
    fbank_feats = []
    spec_feats = []
    spec_phase = []
    trans = []
    phns = []
    wrds = []

    origin_data_dir = os.path.join(timit, mode)
    all_prefix = [file_name.split('.')[0] for file_name in os.listdir(origin_data_dir) if '.wav' in file_name]

    mfcc_dir = os.path.join(root, 'mfcc', mode)
    fbank_dir = os.path.join(root, 'fbank', mode)

    for prefix in all_prefix:
        # mfcc part
        mfcc_path = os.path.join(mfcc_dir, prefix + '.mfcc.cmvn.ark')
        mfcc_feats.append(mfcc_fbank(mfcc_path))
        # fbank part
        fbank_path = os.path.join(fbank_dir, prefix + '.fbank.cmvn.ark')
        fbank_feats.append(mfcc_fbank(fbank_path))
        # spec part
        origin_wav_path = os.path.join(origin_data_dir, prefix + '.wav')
        spec, phase = extract_spec(origin_wav_path)
        spec_feats.append(spec)
        spec_phase.append(phase)
        # check same length
        try:
            assert (len(mfcc_feats[-1]) == len(fbank_feats[-1]) == len(spec_feats[-1]))
        except:
            print('prefix:', prefix)
            print('mfcc length:', len(mfcc_feats[-1]))
            print('fbank length:', len(fbank_feats[-1]))
            print('spec length:', len(spec_feats[-1]))
            sys.exit(-1)

        # transcripts
        origin_txt_path = os.path.join(origin_data_dir, prefix + '.txt')
        trans.append(txt(origin_txt_path))

        length = len(mfcc_feats[-1])
        # phn
        origin_phn_path = os.path.join(origin_data_dir, prefix + '.phn')
        phns.append(phn(origin_phn_path, length))

        # wrd
        origin_wrd_path = os.path.join(origin_data_dir, prefix + '.wrd')
        wrds.append(wrd(origin_wrd_path, length))

    # create meta_data
    meta_data['prefix'] = all_prefix

    save_dir = os.path.join(root, 'processed')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cPickle.dump(meta_data, open(os.path.join(save_dir, f"timit-{mode}-meta.pkl"), 'wb'))
    cPickle.dump(mfcc_feats, open(os.path.join(save_dir, f"timit-{mode}-mfcc.pkl"), 'wb'))
    cPickle.dump(fbank_feats, open(os.path.join(save_dir, f"timit-{mode}-fbank.pkl"), 'wb'))
    cPickle.dump(spec_feats, open(os.path.join(save_dir, f"timit-{mode}-spec.pkl"), 'wb'))
    cPickle.dump(spec_phase, open(os.path.join(save_dir, f"timit-{mode}-spec-phase.pkl"), 'wb'))
    cPickle.dump(trans, open(os.path.join(save_dir, f"timit-{mode}-transcript.pkl"), 'wb'))
    cPickle.dump(phns, open(os.path.join(save_dir, f"timit-{mode}-phn.pkl"), 'wb'))
    cPickle.dump(wrds, open(os.path.join(save_dir, f"timit-{mode}-wrd.pkl"), 'wb'))

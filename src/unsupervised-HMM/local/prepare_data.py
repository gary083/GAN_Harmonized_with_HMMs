import pickle as pkl
import numpy as np
import os,sys
import argparse
## wav.scp , text, utt2spk

def to_4_digits(integer):
    s = str(integer)
    l = len(s)
    for i in range(4-l):
        s = '0' + s
    return "A" + s

def prepare_wav_scp(wav_names, wav_path, tgt_path):
    length = len(wav_names['prefix'])
    wavs = wav_names['prefix']
    with open(tgt_path,'w') as f:
        for i in range(length):
            f.write(to_4_digits(i) + ' ' + os.path.join(wav_path, wavs[i]) + '.wav\n')

def prepare_utt2spk(length, tgt_path):
    with open(tgt_path,'w') as f:
        for i in range(length):
            f.write(to_4_digits(i) + ' ' + to_4_digits(i) + '\n')

def prepare_orc_text(trans, tgt_path):
    length = len(trans)
    with open(tgt_path,'w') as f:
        for i in range(length):
            f.write(to_4_digits(i) + ' ' + ' '.join(trans[i]) + '\n')

def prepare_text_from_output(src_path, tgt_path):
    L = []
    with open(src_path,'r') as f:
        for line in f:
            tokens = line.rstrip().split(' ')
            s = ' '.join(tokens[1:])
            L.append(s)
    with open(tgt_path,'w') as f:
        for i, s in enumerate(L):
            f.write(to_4_digits(i) + ' ' + s + '\n')

def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bnd_type',    type=str, default='orc', help='')
    parser.add_argument('--lm_type',     type=str, default='match', help='')
    parser.add_argument('--data_path',   type=str, default='/home/guanyu/guanyu/handoff/data', help='')
    parser.add_argument('--timit_path',  type=str, default='/home/guanyu/guanyu/timit_data', help='')
    parser.add_argument('--iteration',   type=int, default=1, help='')
    return parser

if __name__ == '__main__':
    parser = addParser()
    args = parser.parse_args()

    data_dir = args.data_path
    train_data_path = data_dir + '/timit_for_HMM/train'
    test_data_path = data_dir + '/timit_for_HMM/test'
    train_data_correct_path = data_dir + '/timit_for_HMM/train_correct'
    train_wavs_names = pkl.load(open(f'{args.data_path}/timit_for_GAN/audio/timit-train-meta.pkl','rb'))
    train_wav_path  = f'{args.timit_path}/train' 
    train_trans_path = f'{args.data_path}/save/{args.bnd_type}_iter{args.iteration}_{args.lm_type}_gan/train_output.txt'
    # train_trans_path = f'{args.data_path}/save/{args.bnd_type}_{args.lm_type}_gan/train_output.txt'

    train_trans = pkl.load(open(f'{args.data_path}/timit_for_GAN/audio/timit-train-phn.pkl','rb'))

    test_wavs_names = pkl.load(open(f'{args.data_path}/timit_for_GAN/audio/timit-test-meta.pkl','rb'))
    test_wav_path  = f'{args.timit_path}/test' 
    test_trans = pkl.load(open(f'{args.data_path}/timit_for_GAN/audio/timit-test-phn.pkl','rb'))

    ## train data
    if not os.path.isdir(train_data_path):
        os.makedirs(train_data_path)
    prepare_text_from_output(train_trans_path, os.path.join(train_data_path,'text'))
    if args.iteration != 1: exit()

    train_lengths = len(train_wavs_names['prefix'])
    prepare_wav_scp(train_wavs_names, train_wav_path, os.path.join(train_data_path,'wav.scp'))
    prepare_utt2spk(train_lengths, os.path.join(train_data_path,'utt2spk'))
    prepare_utt2spk(train_lengths, os.path.join(train_data_path,'spk2utt'))

    ## train correct
    if not os.path.isdir(train_data_correct_path):
        os.makedirs(train_data_correct_path)
    train_lengths = len(train_wavs_names['prefix'])
    prepare_wav_scp(train_wavs_names, train_wav_path, os.path.join(train_data_correct_path,'wav.scp'))
    prepare_utt2spk(train_lengths, os.path.join(train_data_correct_path,'utt2spk'))
    prepare_utt2spk(train_lengths, os.path.join(train_data_correct_path,'spk2utt'))
    prepare_orc_text(train_trans, os.path.join(train_data_correct_path,'text'))

    ## test data
    if not os.path.isdir(test_data_path):
        os.makedirs(test_data_path)
    test_lengths = len(test_wavs_names['prefix'])
    prepare_wav_scp(test_wavs_names, test_wav_path, os.path.join(test_data_path,'wav.scp'))
    prepare_utt2spk(test_lengths, os.path.join(test_data_path,'utt2spk'))
    prepare_utt2spk(test_lengths, os.path.join(test_data_path,'spk2utt'))
    prepare_orc_text(test_trans, os.path.join(test_data_path,'text')) 






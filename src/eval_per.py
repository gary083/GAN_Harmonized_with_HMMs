import math, os, sys, random
import heapq
import argparse
import numpy as np 
import _pickle as pk 

from tqdm import tqdm
from collections import Counter, defaultdict
from eval_bound import *
import os

def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bnd_type',    type=str, default='orc', help='')
    parser.add_argument('--set_type',    type=str, default='test', help='')
    parser.add_argument('--lm_type',     type=str, default='match', help='')
    parser.add_argument('--data_path',   type=str, default='/home/guanyu/guanyu/handoff/data', help='')
    return parser

if __name__ == '__main__':
    parser = addParser()
    args = parser.parse_args()

    # oracle information
    train_phn_path = f'{args.data_path}/timit/audio/timit-{args.set_type}-phn.pkl'
    phone_label = load_pickle(train_phn_path)
    
    # phone information
    lexicon_path = f'{args.data_path}/phones.60-48-39.map.txt'
    phn2idx, idx2phn, phn_mapping = read_phn_map(lexicon_path)

    # put your decoder output here!!
    decode_output_path = f'{args.data_path}/save/{args.bnd_type}_{args.lm_type}/{args.set_type}_output.txt'
    
    new_bound, frame_output, phone_output = read_phn_boundary(decode_output_path)
    print (f'Boundaries:{args.bnd_type} / Setting: {args.lm_type} / Data: {args.set_type}')
    eval_per(phone_output, phone_label, phn_mapping)


    

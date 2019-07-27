import math, os, sys, random
import heapq
import numpy as np 
import _pickle as pk 

from tqdm import tqdm
from collections import Counter, defaultdict
from eval_bound import *
import os

if __name__ == '__main__':
    # oracle information
    typ = sys.argv[1]  # train or test
    train_phn_path = '/groups/public/guanyu/timit_new/audio/timit-{}-phn.pkl'.format(typ)
    phone_label = load_pickle(train_phn_path)
    
    # phone information
    lexicon_path = '/groups/public/wfst_decoder/data/timit_new/phones/phones.60-48-39.map.txt'
    phn2idx, idx2phn, phn_mapping = read_phn_map(lexicon_path)

    # put your decoder output here!!
    decode_dir = sys.argv[2]
    
    # get best wer
    if decode_dir.endswith('txt'):
        decode_output_path = decode_dir
    else:
        wer = os.popen('grep WER %s/wer_* | utils/best_wer.sh' % decode_dir).read()
        _,lmwt,penalty = wer[wer.find('wer'):].rstrip().split('_')
        output_path = os.path.join(decode_dir,'scoring_kaldi/penalty_{}/{}.txt'.format(penalty,lmwt))
        copy_path = os.path.join(decode_dir, 'output.txt')
        
        os.system("cat {} | sort > {}".format(output_path,copy_path))
        print("The result file of decoding corrsponding to the lowest WER is in: {}\n".format(copy_path))
        decode_output_path = os.path.join(decode_dir, 'output.txt')

    new_bound, frame_output, phone_output = read_phn_boundary(decode_output_path)
    
    eval_per(phone_output, phone_label, phn_mapping)
    
    orc_bound_path = '/groups/public/wfst_decoder/data/timit_new/audio/timit-{}-orc-bnd.pkl'.format(typ)
    orc_bound = load_pickle(orc_bound_path)
    frame_label = gen_oracle_frame(orc_bound, phone_label)
    eval_fer(frame_output, frame_label, phn_mapping)


    

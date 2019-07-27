import os, sys
import math
import functools
import numpy as np 
import _pickle as pk 
import argparse

from tqdm import tqdm
from collections import *

def tolerance_precision(bounds, seg_bnds, tolerance_window): 
    #Precision                                                                   
    hit = 0.0                                                                    
    for bound in seg_bnds:                                                       
        for l in range(tolerance_window + 1):                                    
            if (bound + l in bounds) and (bound + l > 0):                        
                hit += 1                                                         
                break                                                            
            elif (bound - l in bounds) and (bound - l > 0):                      
                hit += 1                                                         
                break                                                            
    return (hit / (len(seg_bnds)))                                               
                                                                                 
def tolerance_recall(bounds, seg_bnds, tolerance_window): 
    #Recall                                                                      
    hit = 0.0                                                                    
    for bound in bounds:                                                         
        for l in range(tolerance_window + 1):                                    
            if (bound + l in seg_bnds) and (bound + l > 0):                      
                hit += 1                                                         
                break                                                            
            elif (bound - l in seg_bnds) and (bound - l > 0):                    
                hit += 1                                                         
                break                                                            
    return (hit / (len(bounds)))      

def get_bound_score(bounds, seg_bnds, tolerance_window):
    precision = tolerance_precision(bounds, seg_bnds, tolerance_window)
    recall    = tolerance_recall(bounds, seg_bnds, tolerance_window)
    r_score, f_score = get_rf_score(100*precision, 100*recall)
    return r_score
    # return precision, recall, r_score, f_score

def get_rf_score(precision, recall):
    if recall == 0 and precision == 0:
        f_score = -1
        r_score = -1
    else:
        f_score = 2 * precision * recall / (precision + recall)
        os_score = (recall / precision - 1) * 100
        r_score = 1 - (abs(math.sqrt((100 - recall) * (100 - recall) + os_score ** 2)) + abs((recall - 100 - os_score) / math.sqrt(2))) / 200
    return r_score * 100, f_score

def print_bound_score(p_list, r_list, data_type):
    precision = 100* sum(p_list) / len(p_list)
    recall    = 100* sum(r_list) / len(r_list)
    r_score, f_score = get_rf_score(precision, recall)
    print (f'      GAS Boundaries({data_type}) - precision: {precision:.2f}, recall: {recall:.2f}, f_score: {f_score:.2f}, r_score: {r_score:.2f}')

def read_phn_48_map(path):
    all_lines = open(path, 'r').read().splitlines()
    phn_mapping = {}
    for line in all_lines:
        if line.strip() == "":
            continue
        phn_mapping[line.split()[0]] = line.split()[1]

    all_48_phn = list(set(phn_mapping.values()))
    assert(len(all_48_phn) == 48)
    phn2idx_48 = dict(zip(all_48_phn, range(len(all_48_phn))))
    idx2phn_48 = dict(zip(range(len(all_48_phn)), all_48_phn))

    return phn_mapping, idx2phn_48, phn2idx_48

def load_oracle(oracle_data, data_dir):
    phn_60_to_48, idx2phn, phn2idx = read_phn_48_map(f'{data_dir}/phones.60-48-39.map.txt')
    oracle_bound, oracle_phn_seq = [], []

    for line in oracle_data:
        bnd, phn = [], []
        for tup in line:
            bnd.append(tup[1])
            phn.append(phn_60_to_48[tup[0]])
        bnd.append(line[-1][2])
        oracle_bound.append(bnd)
        oracle_phn_seq.append(phn)
    return oracle_bound, oracle_phn_seq

def output_text(file_name, text):
    with open(file_name, 'w') as fout:
        for sentence in text:
            fout.write(' '.join(sentence)+'\n')

def process_data(gas_dir, timit_dir, save_dir, data_type='train', show_bnd_score=True, max_length=400, split_num=3000):
    # Load GAS data 
    gas_meta  = pk.load(open(f'{gas_dir}/gas-{data_type}-meta.pkl', 'rb'))
    gas_bound = pk.load(open(f'{gas_dir}/gas-{data_type}-bnd.pkl', 'rb'))
    gas_prob  = pk.load(open(f'{gas_dir}/gas-{data_type}-prob.pkl', 'rb'))

    # Load TIMIT feature
    timit_meta = pk.load(open(f'{timit_dir}/timit-{data_type}-meta.pkl', 'rb'))
    timit_phn  = pk.load(open(f'{timit_dir}/timit-{data_type}-phn.pkl', 'rb'))
    timit_wav  = pk.load(open(f'{timit_dir}/timit-{data_type}-mfcc-nor.pkl', 'rb'))

    # Process TIMIT oracle data
    timit_oracle_bound, timit_oracle_phn = load_oracle(timit_phn, save_dir)

    # Initial output data
    meta = defaultdict(list) 
    gas_bnd, orc_bnd = [], []
    phn, length, wav, gas = [], [], [], []
    if show_bnd_score: precision_list, recall_list = [], []

    # Start processing...
    for idx, prefix in enumerate(timit_meta['prefix']):
        gas_bnd_idx = gas_meta['prefix'].index(prefix)
        assert (len(timit_oracle_bound[idx]) == len(timit_oracle_phn[idx])+1)

        # Training set selection
        if data_type == 'train' and len(timit_wav[idx]) > max_length: 
            continue

        # Select output data
        meta['prefix'].append(prefix)
        gas.append(gas_prob[gas_bnd_idx])
        gas_bnd.append(gas_bound[gas_bnd_idx])
        orc_bnd.append(timit_oracle_bound[idx])
        phn.append(timit_oracle_phn[idx])
        length.append(len(timit_wav[idx]))
        wav.append(timit_wav[idx])
        if show_bnd_score:
            precision_list.append(tolerance_precision(timit_oracle_bound[idx], gas_bound[gas_bnd_idx], 2))
            recall_list.append(tolerance_recall(timit_oracle_bound[idx], gas_bound[gas_bnd_idx], 2))

    if show_bnd_score: print_bound_score(precision_list, recall_list, data_type)

    # Build processed directory
    result_directory = f'{save_dir}/result'
    audio_directory  = f'{save_dir}/timit_for_GAN/audio'
    text_directory   = f'{save_dir}/timit_for_GAN/text'

    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    if not os.path.exists(audio_directory):
        os.makedirs(audio_directory)
    if not os.path.exists(text_directory):
        os.makedirs(text_directory)

    # Save as pickle file
    pk.dump(meta,                open(f'{audio_directory}/timit-{data_type}-meta.pkl'      , 'wb'))
    pk.dump(np.array(gas_bnd),   open(f'{audio_directory}/timit-{data_type}-uns1-bnd.pkl'   , 'wb'))
    pk.dump(np.array(orc_bnd),   open(f'{audio_directory}/timit-{data_type}-orc1-bnd.pkl'   , 'wb'))
    pk.dump(np.array(phn),       open(f'{audio_directory}/timit-{data_type}-phn.pkl'       , 'wb'))
    pk.dump(np.array(length),    open(f'{audio_directory}/timit-{data_type}-length.pkl'    , 'wb'))
    pk.dump(np.array(wav),       open(f'{audio_directory}/timit-{data_type}-mfcc-nor.pkl'  , 'wb'))
    pk.dump(np.array(gas),       open(f'{audio_directory}/timit-{data_type}-gas.pkl'       , 'wb'))

    # output lm data
    if data_type == 'train':
        output_text(f'{text_directory}/match_lm.48',    phn)
        output_text(f'{text_directory}/nonmatch_lm.48', phn[split_num:])

def argParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gas_root',       type=str, help='')
    parser.add_argument('--timit_root',     type=str, help='')
    parser.add_argument('--save_root',      type=str, help='')
    parser.add_argument('--show_bnd_score', action='store_true', help='')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # gas_root     = sys.argv[1]
    # timit_root   = sys.argv[2] 
    # save_root    = sys.argv[3]
    # show_bnd_score     = sys.argv[4]
    args = argParser()
    process_data(args.gas_root, args.timit_root, args.save_root, data_type='train', show_bnd_score=args.show_bnd_score)
    process_data(args.gas_root, args.timit_root, args.save_root, data_type='test', show_bnd_score=args.show_bnd_score)
    # process_train(gas_root, feature_root, save_root)
    # process_dev(gas_root, feature_root, save_root)

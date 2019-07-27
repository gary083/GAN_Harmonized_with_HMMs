import math, os, sys, random
import heapq
import numpy as np 
import _pickle as pk 

from tqdm import tqdm
from collections import Counter, defaultdict

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
    
def get_rf_score(precision, recall):
    if recall == 0 and precision == 0:
        f_score = -1
        r_score = -1
    else:
        f_score = 2 * precision * recall / (precision + recall)
        os_score = (recall / precision - 1) * 100
        r_score = 1 - (abs(math.sqrt((100 - recall) * (100 - recall) + os_score ** 2)) + abs((recall - 100 - os_score) / math.sqrt(2))) / 200
    return r_score * 100, f_score

def read_result(result_path):
    return [line.strip().split()[1:] for line in open(result_path, 'r')]

def load_pickle(file_name):
    return pk.load(open(file_name, 'rb'))

def read_phn_map(path):
    all_lines = open(path, 'r').read().splitlines()
    temp_phn_mapping = {}
    for line in all_lines:
        if line.strip() == "":
            continue
        temp_phn_mapping[line.split()[1]] = line.split()[2]

    all_phn = list(temp_phn_mapping.keys())
    assert(len(all_phn) == 48)
    phn2idx = dict(zip(all_phn, range(len(all_phn))))
    idx2phn = dict(zip(range(len(all_phn)), all_phn))
    phn_size = len(all_phn) 
    phn_mapping = {}
    for phn in all_phn:
        phn_mapping[phn] = temp_phn_mapping[phn]
    return phn2idx, idx2phn, phn_mapping

def convert_39(phn, phn_mapping):
    return [phn_mapping[p] for p in phn]

def edit_distance(seq1, seq2):
    if len(seq1) > len(seq2):
        seq1, seq2 = seq2, seq1

    distances = range(len(seq1) + 1)
    for i2, c2 in enumerate(seq2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(seq1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min(distances[i1], distances[i1 + 1], distances_[-1]))
        distances = distances_
    return distances[-1]

def read_phn_boundary(map_path):
    all_lines = open(map_path, 'r').read().splitlines()
    new_bnd = []
    frame_output = []
    phone_output = []
    for line in all_lines:
        bnd = []
        phone = []
        frame = [] 
        prev = ''
        for i, p in enumerate(line.strip().split(' ')[1:]):
            if '_' in p:
                p = p.split('_')[1]
            frame.append(p)
            if p != prev:
                bnd.append(i)
                phone.append(p)
            prev = p
        bnd.append(len(line.strip().split(' ')[1:])-1)
        new_bnd.append(bnd)
        frame_output.append(frame)
        phone_output.append(phone)
    return new_bnd, frame_output, phone_output

def gen_oracle_frame(orc_bound, orc_phn):
    frame_label = []
    for bnd, phn in zip(orc_bound, orc_phn):
        assert (len(bnd)==len(phn)+1)
        frame = []
        prev_b = 0
        for b, p in zip(bnd[1:], phn):
            frame.extend([p]*(b - prev_b))
            prev_b = b
        frame.extend([phn[-1]])
        frame_label.append(frame)
    return frame_label

def eval_bnd(new_bound, orc_bound):
    precision_list = []
    recall_list = []
    for new_bnd, orc_bnd in zip(new_bound, orc_bound):
        precision_list.append(tolerance_precision(orc_bnd, new_bnd, 2))
        recall_list.append(tolerance_recall(orc_bnd, new_bnd, 2))
    precision = 100 * sum(precision_list) / len(precision_list)
    recall    = 100 * sum(recall_list) / len(recall_list)
    r_score, f_score = get_rf_score(precision, recall)
    print (f'Bound Eval: ')
    print (f'Precision: {precision} Recall: {recall} F_score: {f_score} R_score: {r_score}')

def eval_per(phone_output, oracle_output, phn_mapping):
    total_wer = 0
    total_length = 0
    for phn, orc in zip(phone_output, oracle_output):
        wer = edit_distance(convert_39(phn, phn_mapping), convert_39(orc, phn_mapping))
        total_wer += wer 
        total_length += len(orc)
    print (f'Phoneme error rate: {total_wer/total_length*100}')

def eval_fer(frame_output, oracle_output, phn_mapping):
    total_wer = 0
    total_length = 0
    for frame, orc in tqdm(zip(frame_output, oracle_output)):
        wer = edit_distance(convert_39(frame, phn_mapping), convert_39(orc, phn_mapping))
        total_wer += wer 
        total_length += len(orc)
    print (f'Frame error rate: {total_wer/total_length*100}')

def eval_fer2(frame_output, oracle_output, phn_mapping):
    total_wer = 0
    total_length = 0
    for frame, orc in tqdm(zip(frame_output, oracle_output)):
        total_wer += cal_frame_error(convert_39(frame, phn_mapping), convert_39(orc, phn_mapping))
        total_length += len(orc)
    print (f'Frame error rate: {total_wer/total_length*100}')

def cal_frame_error(frame,orc):
    wrong = 0
    for x,y in zip(frame,orc):
        if x != y :
            wrong += 1
    return wrong

if __name__ == '__main__':
    # oracle information
    typ = 'train'
    train_phn_path = '/groups/public/wfst_decoder/data/timit_new/audio/timit-{}-phn.pkl'.format(typ)
    orc_bound_path = '/groups/public/wfst_decoder/data/timit_new/audio/timit-{}-orc-bnd.pkl'.format(typ)
    
    # uns_bound_path = 'timit/audio/timit-train-uns-bnd.pkl'
    # phone information
    lexicon_path = '/groups/public/wfst_decoder/data/timit_new/phones/phones.60-48-39.map.txt'

    # put your decoder output here!!
    # output format: b'0000' sil sil sil sil sil sil sil sil sil ix ix ix ix ix ...
    #decode_output_path = 'exp/uns_matched_gas/decode_gas/phones_ali.txt'
    decode_output_path = sys.argv[1]

    phone_label = load_pickle(train_phn_path)
    orc_bound = load_pickle(orc_bound_path)
    # uns_bound = load_pickle(uns_bound_path)	
    phn2idx, idx2phn, phn_mapping = read_phn_map(lexicon_path)

    new_bound, frame_output, phone_output = read_phn_boundary(decode_output_path)
    frame_label = gen_oracle_frame(orc_bound, phone_label)
    eval_bnd(new_bound, orc_bound)
    # eval_bnd(uns_bound, orc_bound)
    eval_per(phone_output, phone_label, phn_mapping)
    #eval_fer(frame_output, frame_label, phn_mapping)


    

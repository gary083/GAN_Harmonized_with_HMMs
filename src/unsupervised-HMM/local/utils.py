import numpy as np 
import math
def read_phone_txt(txt_path, dim = 1):
    phone_list = []
    with open(txt_path,'r') as f:
        for line in f:
            phone = line.rstrip().split()[dim]
            if phone not in phone_list:
                phone_list.append(phone)
    return phone_list

def write_phone_file(phone_list,txt_path, inv = False):
    with open(txt_path,'w') as f:
        if inv :
            for idx, phone in enumerate(phone_list):
                f.write("{} {}\n".format(phone,idx))
        else:
            for idx, phone in enumerate(phone_list):
                f.write("{} {}\n".format(idx,phone))

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def smooth(y, box_pts, ratio):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    y_smooth = y * ratio + y_smooth * (1-ratio)
    return y_smooth

def to_keep_prob(gas_raw, lengths, scale = 500, smooth_len = 3, smooth_ratio = 0.5):
    keep_probs = np.zeros(gas_raw.shape)
    for idx, (l, gas) in enumerate(zip(lengths, gas_raw)):
        keep_raw = gas[1:] - gas[:-1] 
        keep_prob = np.insert(keep_raw,0, -0.1)
        for n in range(l-2):
            if keep_prob[n+1] <= keep_prob[n] or keep_prob[n+1] <= keep_prob[n+2]:
                keep_prob[n+1] = -0.1
        keep_prob *= scale
        keep_prob = sigmoid(keep_prob)
        
        if smooth_len > 1:
            keep_prob = smooth(keep_prob, smooth_len, smooth_ratio)

        keep_probs[idx] = keep_prob
    
    return keep_probs
        



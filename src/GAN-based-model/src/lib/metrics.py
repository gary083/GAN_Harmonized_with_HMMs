import torch

import editdistance as ed
from itertools import groupby
from typing import List


def get_phoneseq(frame_labels):
    phone_seq = [key for key, group in groupby(frame_labels)]
    return phone_seq

def calc_fer(prediction : List[torch.Tensor],
            ground_truth : List[torch.Tensor]) -> float :

    frame_error = 0
    frame_num = 0
    for p, g in zip(prediction, ground_truth):
        l = g.shape[0]
        p = p[:l]
        frame_error += sum(p!=g)
        frame_num += l
    return frame_error, frame_num

def calc_per(prediction : List[torch.Tensor],
            ground_truth : List[torch.Tensor]) -> float :

    prediction = [get_phoneseq(p) for p in prediction]
    ground_truth = [get_phoneseq(p) for p in ground_truth]
    eds = [ed.eval(p, l) / len(l) for p, l in zip(prediction, ground_truth)]
    return sum(eds) / (len(eds) + 1e-6), eds

def calc_acc(prediction : List[torch.Tensor],
            ground_truth : List[torch.Tensor]) -> float :
    acc = []
    for p, g in zip(prediction, ground_truth):
        l = g.shape[0]
        p = p[:l]
        acc.append(sum(p==g)/l)
    return sum(acc) / (len(acc) + 1e-10)


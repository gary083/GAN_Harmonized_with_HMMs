import torch
import numpy as np


def get_tensor_from_array(arr: np.array) -> torch.Tensor:
    arr = torch.Tensor(arr)
    if torch.cuda.is_available():
        arr = arr.cuda()
    return arr

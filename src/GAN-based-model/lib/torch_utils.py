import torch
import numpy as np

epsilon = 1e-8


def get_tensor_from_array(arr: np.array) -> torch.Tensor:
    arr = torch.Tensor(arr)
    if torch.cuda.is_available():
        arr = arr.cuda()
    return arr


def masked_reduce_mean(arr: torch.Tensor, mask: torch.Tensor):
    assert(arr.shape[:2] == mask.shape[:2])
    if arr.ndimension() == 3:
        mask = mask.unsqueeze(2)
    arr = arr * mask
    return arr.sum(dim=1) / (mask.sum(dim=1) + epsilon)


def masked_reduce_sum(arr: torch.Tensor, mask: torch.Tensor):
    assert(arr.shape[:2] == mask.shape[:2])
    if arr.ndimension() == 3:
        mask = mask.unsqueeze(2)
    arr = arr * mask
    return arr.sum(dim=1)


def l2_loss(pred, tar, mask):
    assert(pred.shape == tar.shape)
    mask_l2_loss = torch.sum((pred - tar) ** 2, dim=2)  # shape: (N, T)
    loss = torch.mean(masked_reduce_mean(mask_l2_loss, mask))
    return loss


def cpc_loss(pred, tar, pred_mask, attention_mask, mode=''):
    """
    :param pred: shape: (N, T, V)
    :param tar:  shape: (N, L, V)
    :param pred_mask: shape: (N, T), ones at where to predict
    :param attention_mask: shape: (N, T), ones at non-padding timesteps
    assert T = L
    """
    assert(pred.shape == tar.shape)
    if mode == 'same':
        inner_products = torch.einsum('nte,nle->ntl', pred, tar)  # shape: (N, T, L)
        exp_inner_products = torch.exp(inner_products)
        denom = masked_reduce_mean(exp_inner_products, attention_mask)  # shape: (N, L)
        num = torch.diagonal(inner_products, dim1=1, dim2=2)  # shape: (N, L)
        cpc = -torch.mean(masked_reduce_mean(
            num - torch.log(denom + epsilon),
            pred_mask,
        ))
    else:
        pos_inner_products = torch.einsum('nte,nte->nt', pred, tar)  # shape: (N, T)
        neg_inner_products = torch.einsum('nte,nle->nlt', pred, torch.flip(tar, dims=[0]))  # shape: (N, T, L)
        max_logit = max(torch.max(pos_inner_products).item(), torch.max(neg_inner_products).item())
        pos_inner_products = pos_inner_products - max_logit
        neg_inner_products = neg_inner_products - max_logit

        exp_pos_inner_products = torch.exp(pos_inner_products)
        sum_exp_neg_inner_products = masked_reduce_sum(
            torch.exp(neg_inner_products),
            torch.flip(attention_mask, dims=[0]),
        )  # shape: (N, T)

        cpc = torch.mean(masked_reduce_mean(
            torch.log(1 + sum_exp_neg_inner_products / (exp_pos_inner_products + epsilon)),
            pred_mask,
        ))
    return cpc


def intra_segment_loss(logits, repeats, mask, sep_size):
    probs = torch.softmax(logits, dim=-1)
    start_prob = probs[:sep_size]
    end_prob = probs[sep_size:]
    partial_mask = mask[:sep_size]
    error = torch.sum((start_prob - end_prob) ** 2, dim=-1)
    return torch.sum(masked_reduce_sum(error, partial_mask)) / (torch.sum(repeats) + epsilon)


def inter_segment_loss(logits, mask):
    """
    Implement Jensen-Shannon Divergence for multiple distributions
    JSD = H(sum(Pi)/m) - H(Pi)/m
    """
    probs = torch.softmax(logits, dim=-1)  # (N, T, V)
    mean_probs = masked_reduce_mean(probs, mask)  # (N, V)
    H_of_mean_probs = torch.sum(-mean_probs * torch.log(mean_probs + epsilon), dim=-1)  # (N,)

    Hs = torch.sum(-probs * torch.log(probs + epsilon), dim=-1)  # (N, T)
    mean_of_Hs = masked_reduce_mean(Hs, mask)  # (N,)
    JSDs = H_of_mean_probs - mean_of_Hs
    JSD = torch.mean(JSDs)
    return -JSD


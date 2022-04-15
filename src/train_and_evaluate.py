import copy
import math
import random

import numpy
import torch
from torch import nn

from src.expressions_transfer import *

USE_CUDA = torch.cuda.is_available()
MAX_OUTPUT_LENGTH = 33


def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):
    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    length = length - 1
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.reshape(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = torch.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.reshape(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()

    # max_len = losses.size(1)
    #
    # losses = losses.sum(dim=1) * (length.float() / max_len)

    loss = losses.sum() / length.float().sum()

    # if loss.item() > 10:
    #     print(losses, target)
    return loss


def train(input_batch, input_length, output_batch, output_length, num_stack_batch, output_lang, num_pos,
          model, optimizer, teacher_forcing):
    model.train()

    optimizer.zero_grad()

    input_var = torch.LongTensor(input_batch).transpose(0, 1)

    trg = torch.LongTensor(output_batch).transpose(0, 1)
    if USE_CUDA:
        input_var = input_var.cuda()
        trg = trg.cuda()

    output = model.forward(input_var, input_length, trg, num_pos, output_lang, teacher_forcing, num_stack_batch)
    trg = trg[1:]
    loss = masked_cross_entropy(output.transpose(0, 1), trg.transpose(0, 1), output_length)

    loss.backward()
    # print([x.grad for x in optimizer.param_groups[0]['params']])
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    optimizer.step()

    return loss.item()


def train_wl(input_batch, input_length, output_batch, output_length, num_stack_batch, output_lang, num_pos,
             model, optimizer, teacher_forcing, a=0.1):
    model.train()

    optimizer.zero_grad()

    input_var = torch.LongTensor(input_batch).transpose(0, 1)

    trg = torch.LongTensor(output_batch).transpose(0, 1)
    if USE_CUDA:
        input_var = input_var.cuda()
        trg = trg.cuda()

    output = model.forward(input_var, input_length, trg, num_pos, output_lang, teacher_forcing, num_stack_batch)
    trg = trg[1:]
    ce = masked_cross_entropy(output.transpose(0, 1), trg.transpose(0, 1), output_length)
    wl_mse = a * model.encoder.wl.MSELoss()
    loss = ce + wl_mse
    loss.backward()
    # print([x.grad for x in optimizer.param_groups[0]['params']])
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    optimizer.step()

    return loss.item(), ce.item(), wl_mse.item()


def evaluate(input_batch, input_length, output_batch, output_lang, num_pos, model):
    model.eval()

    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    trg = None
    if output_batch != None:
        trg = torch.LongTensor(output_batch).transpose(0, 1)
    if USE_CUDA:
        input_var = input_var.cuda()
        if trg != None:
            trg = trg.cuda()
    output = model.forward(input_var, input_length, trg, num_pos, output_lang, -1)

    output = torch.argmax(output.transpose(0, 1), dim=-1)
    if USE_CUDA:
        output = output.cpu()

    return output.detach().numpy()


def evaluate_with_attn(input_batch, input_length, output_batch, output_lang, num_pos, model):
    model.eval()

    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    trg = None
    if output_batch != None:
        trg = torch.LongTensor(output_batch).transpose(0, 1)
    if USE_CUDA:
        input_var = input_var.cuda()
        if trg != None:
            trg = trg.cuda()
    output, gate, attn = model.forward_with_attn(input_var, input_length, None, num_pos, output_lang)
    output = torch.argmax(output.transpose(0, 1), dim=-1)

    if USE_CUDA:
        output = output.cpu()
        gate = gate.cpu()
        attn = attn.cpu()

    return output.detach().numpy(), gate.detach().numpy(), attn.detach().numpy()

def evaluate_with_beam_search(input_batch, input_length, output_batch, output_lang, num_pos, model):
    model.eval()

    input_var = torch.LongTensor(input_batch).transpose(0, 1)

    if USE_CUDA:
        input_var = input_var.cuda()

    output = model.beam_search(input_var, input_length, num_pos, output_lang)

    return output



def compute_prefix_tree_result(test_res, test_tar, output_lang, num_list, num_stack=None):
    # print(test_res, test_tar)
    temp = []
    for i in test_res:
        word = output_lang.index2word[i]
        if word == 'EOS':
            break
        if word == 'SOS':
            temp.reverse()
            break
        temp.append(i)
    test_res = temp

    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar[1:], output_lang, num_list, num_stack)

    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar

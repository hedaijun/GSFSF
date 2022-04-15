import math
import random
from copy import deepcopy
from typing import Optional

import numpy
import torch
from torch import Tensor
from torch import nn
from transformers import BertModel, BertConfig, AutoModel

USE_CUDA = torch.cuda.is_available()
MAX_OUTPUT_LENGTH = 50


class Beam:  # the class save the beam node
    def __init__(self, score, input_var, hidden, feed_hidden, all_output):
        self.score = score
        self.input_var = input_var
        self.hidden = hidden
        self.feed_hidden = feed_hidden
        self.all_output = all_output


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, model_name, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = AutoModel.from_pretrained(model_name)
        self.trans = nn.Linear(hid_dim, hid_dim)
        self.norm = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, input_length):
        seq_mask = []
        max_len = max(input_length)
        for i in input_length:
            seq_mask.append([1 for _ in range(i)] + [0 for _ in range(i, max_len)])
        seq_mask = torch.FloatTensor(seq_mask)
        if USE_CUDA:
            seq_mask = seq_mask.cuda()
        embedded = self.embedding(src.transpose(0, 1), attention_mask=seq_mask, return_dict=True)[0].transpose(0, 1)
        output = self.norm(self.trans(embedded))
        feed_hidden = output[0]

        return output, feed_hidden

class WhiteningLayer(nn.Module):
    def __init__(self, hidden_size):
        super(WhiteningLayer, self).__init__()
        self.hidden_size = hidden_size

        self.bias = nn.Parameter(torch.randn(hidden_size))
        self.kernel = nn.Parameter(torch.randn(hidden_size, hidden_size))
        nn.init.normal_(self.bias, mean=0.0, std=1 / math.sqrt(hidden_size))
        nn.init.normal_(self.kernel, mean=0.0, std=1 / math.sqrt(hidden_size))

    def forward(self, query, feed_hiddens, values):
        max_len = feed_hiddens.size(0)
        batch_size = feed_hiddens.size(1)

        q = query.view(1, batch_size, self.heads, self.dk).permute(1, 2, 0, 3).repeat(1, 1, max_len, 1)
        k = feed_hiddens.view(max_len, batch_size, self.heads, self.dk).permute(1, 2, 0, 3)
        v = values.view(max_len, batch_size, self.heads, self.dk).permute(1, 2, 0, 3)

        energy = torch.relu(torch.matmul(torch.cat([q, k], dim=-1), self.attn) + self.bias)
        attn = torch.matmul(energy, self.score).transpose(2, 3)
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v).view(batch_size, self.hidden_size).unsqueeze(0)

        return out

def whitening(anchor, pos, neg):
    overall = torch.cat([anchor.detach(), pos.detach(), neg.detach()], dim=0)
    factor = 1. / (overall.size(0) - 1)
    m = overall.mean(dim=0, keepdim=True)
    tmp = overall - m
    cov = torch.mm(tmp.t(), tmp) * factor
    U, S, V = cov.svd()

    S = torch.diag_embed(1 / torch.sqrt(S))

    W = torch.mm(U, S)

    anchor = torch.mm(anchor - m, W)
    pos = torch.mm(pos - m, W)
    neg = torch.mm(neg - m, W)

    return anchor, pos, neg

class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.gelu = torch.nn.GELU()
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.relu(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill_(num_mask, -1e12)
        return score


class SerialTFMLayer(nn.TransformerDecoderLayer):
    def __init__(self, *args, **kargs):
        super(SerialTFMLayer, self).__init__(*args, **kargs)

    def Serialforward(self, tgt: Tensor, history: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                      memory_mask: Optional[Tensor] = None,
                      tgt_key_padding_mask: Optional[Tensor] = None,
                      memory_key_padding_mask: Optional[Tensor] = None):
        tgt2 = self.self_attn(tgt, history, history, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


class SerialTFM(nn.TransformerDecoder):
    def __init__(self, hidden_size, *args, **kargs):
        super(SerialTFM, self).__init__(*args, **kargs)
        pe = torch.zeros(MAX_OUTPUT_LENGTH, hidden_size)
        position = torch.arange(0, MAX_OUTPUT_LENGTH).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) *
                             -(math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def Serialforward(self, tgt: Tensor, history: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                      memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                      memory_key_padding_mask: Optional[Tensor] = None):

        if history.size(0) > 0:
            output = tgt + self.pe[history.size(1)]
        else:
            output = tgt + self.pe[0]

        out_list = []
        for id, mod in enumerate(self.layers):
            if history.size(0) > 0:
                out_list.append(output)
                output = mod.Serialforward(output, torch.cat([history[id], output], dim=0),
                                           memory, tgt_mask=tgt_mask,
                                           memory_mask=memory_mask,
                                           tgt_key_padding_mask=tgt_key_padding_mask,
                                           memory_key_padding_mask=memory_key_padding_mask)
            else:
                out_list.append(output)
                output = mod.Serialforward(output, output,
                                           memory, tgt_mask=tgt_mask,
                                           memory_mask=memory_mask,
                                           tgt_key_padding_mask=tgt_key_padding_mask,
                                           memory_key_padding_mask=memory_key_padding_mask)

        return output, torch.stack(out_list, dim=0)


class Decoder(nn.Module):
    def __init__(self, hidden_size, n_layers, dropout=0.5):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        decoder_layer = SerialTFMLayer(768, 12, dim_feedforward=768 * 4)
        self.tfm_dec = SerialTFM(hidden_size, decoder_layer, n_layers)
        print('decoder layer: ', n_layers)

    def forward(self, input_list, history, encoder_outputs, seq_mask):
        output, new_history = self.tfm_dec.Serialforward(input_list, history,
                                                         memory=encoder_outputs, memory_key_padding_mask=seq_mask)

        return output, new_history


def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, hidden_size):
    indices = list()
    sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices)
    masked_index = torch.BoolTensor(masked_index)
    masked_index = masked_index.view(batch_size, num_size, hidden_size)
    if USE_CUDA:
        indices = indices.cuda()
        masked_index = masked_index.cuda()
    all_outputs = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    return all_num.masked_fill_(masked_index, 0.0)


def generate_decoder_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    if USE_CUDA:
        decoder_output = decoder_output.cpu()
    for i in range(target.size(0)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
    return target


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, stuff_size, hidden_size, output_lang, beam_size=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.stuff_size = stuff_size
        self.output_lang = output_lang
        self.beam_size = beam_size

        self.embedding_weight = torch.nn.Parameter(torch.randn((stuff_size, hidden_size)))
        self.score = Score(hidden_size, hidden_size)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, input_length, trg, num_pos, output_lang, teacher_forcing, num_stack=None):
        if num_stack is not None:
            num_stack = deepcopy(num_stack)

        seq_mask = []
        max_len = max(input_length)
        for i in input_length:
            seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
        seq_mask = torch.BoolTensor(seq_mask)

        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        num_mask = []
        for pos in num_pos:
            num_mask.append(
                [0] * self.stuff_size + [0 for _ in range(len(pos))] + [1 for _ in range(len(pos), num_size)])
        num_mask = torch.BoolTensor(num_mask)

        if USE_CUDA:
            seq_mask = seq_mask.cuda()
            num_mask = num_mask.cuda()
        assert len(num_pos) == src.size(1)
        batch_size = src.size(1)
        if trg is not None:
            max_len = trg.shape[0]
        else:
            max_len = MAX_OUTPUT_LENGTH

        encoder_out, problem = self.encoder.forward(src, input_length)

        # make output word dict
        word_dict_vec = self.embedding_weight.unsqueeze(0).repeat(batch_size, 1, 1)

        num_embedded = get_all_number_encoder_outputs(encoder_out, num_pos, batch_size, num_size, self.hidden_size)

        word_dict_vec = torch.cat((word_dict_vec, num_embedded), dim=1)

        out_list = []
        input_ids = torch.LongTensor([output_lang.word2index["SOS"]] * batch_size)
        if USE_CUDA:
            input_ids = input_ids.cuda()
        trg_one_hot = torch.nn.functional.one_hot(input_ids, num_classes=word_dict_vec.size(1)).float()
        input_num = trg_one_hot.unsqueeze(1).bmm(word_dict_vec).transpose(0, 1)
        history = torch.FloatTensor([])
        if USE_CUDA:
            history = history.cuda()
        if teacher_forcing > random.random():
            for t in range(0, max_len - 1):
                output, new_history = self.decoder.forward(input_num, history, encoder_out, seq_mask)

                score_list = self.score(output[-1].unsqueeze(1), word_dict_vec, num_mask)

                out_list.append(score_list)

                input_ids = generate_decoder_input(trg[t + 1], score_list, num_stack, self.stuff_size,
                                                   output_lang.word2index["UNK"])

                trg_one_hot = torch.nn.functional.one_hot(input_ids, num_classes=word_dict_vec.size(1)).float()
                input_num = trg_one_hot.unsqueeze(1).bmm(word_dict_vec).transpose(0, 1)
                history = torch.cat([history, new_history], dim=1)

            return torch.stack(out_list, dim=0)
        else:
            for t in range(0, max_len - 1):
                output, new_history = self.decoder.forward(input_num, history, encoder_out, seq_mask)

                score_list = self.score(output[-1].unsqueeze(1), word_dict_vec, num_mask)

                out_list.append(score_list)

                input_ids = torch.argmax(score_list, dim=-1)

                trg_one_hot = torch.nn.functional.one_hot(input_ids, num_classes=word_dict_vec.size(1)).float()
                input_num = trg_one_hot.unsqueeze(1).bmm(word_dict_vec).transpose(0, 1)
                history = torch.cat([history, new_history], dim=1)

            decoder_out = torch.stack(out_list, dim=0)

            return decoder_out

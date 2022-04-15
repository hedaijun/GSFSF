import math
import random
from copy import deepcopy

import numpy
import torch
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


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size

        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.factor = 1. / math.sqrt(hidden_size)

    def forward(self, hidden, encoder_outputs, pe=None, seq_mask=None, dropout=None):
        max_len = encoder_outputs.size(0)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        # For each position of encoder outputs
        this_batch_size = encoder_outputs.size(1)

        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, 2 * self.hidden_size)

        attn_energies = (self.factor * math.log(max_len)) * self.score(torch.relu(self.attn(energy_in)))  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if pe is not None:
            attn_energies = attn_energies + pe
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
        attn_energies = self.softmax(attn_energies)
        if dropout is not None:
            attn_energies = dropout(attn_energies)
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return attn_energies.unsqueeze(1)


class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
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


class Decoder(nn.Module):
    def __init__(self, hidden_size, n_layers=2, dropout=0.5):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.dropout = nn.Dropout(0.5)

        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers)

        self.attn = Attn(hidden_size)

        print('decoder layer: ', n_layers)

    def forward(self, input_ids, hidden, word_dict_vec, encoder_outputs, seq_mask):
        trg_one_hot = nn.functional.one_hot(input_ids, num_classes=word_dict_vec.size(1)).float()
        last_embedded = trg_one_hot.unsqueeze(1).bmm(word_dict_vec).transpose(0, 1)

        out, hidden = self.gru(last_embedded, hidden)

        attn_weights = self.attn(out, encoder_outputs, seq_mask=seq_mask)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)).transpose(0, 1)

        return torch.cat((out, context), dim=-1).squeeze(0), hidden


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

        self.embedding_weight = nn.Parameter(torch.randn((stuff_size, hidden_size)))
        self.init = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

        self.score = Score(hidden_size * 2, hidden_size)
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
        hidden = problem.unsqueeze(0).repeat(self.decoder.n_layers, 1, 1)

        input = torch.LongTensor([output_lang.word2index["SOS"]] * batch_size)
        if USE_CUDA:
            input = input.cuda()
        if teacher_forcing > random.random():
            for t in range(0, max_len - 1):
                output, hidden = self.decoder.forward(input, hidden,
                                                           word_dict_vec, encoder_out, seq_mask)
                score_list = self.score(output.unsqueeze(1), word_dict_vec, num_mask)

                out_list.append(score_list)

                input = generate_decoder_input(trg[t + 1], score_list, num_stack, self.stuff_size,
                                               output_lang.word2index["UNK"])
                trg[t + 1] = input
            decoder_out = torch.stack(out_list, dim=0)
            return decoder_out
        else:
            for t in range(0, max_len - 1):
                output, hidden = self.decoder.forward(input, hidden,
                                                           word_dict_vec, encoder_out, seq_mask)

                score_list = self.score(output.unsqueeze(1), word_dict_vec, num_mask)

                out_list.append(score_list)

                input = torch.argmax(score_list, dim=-1)

            decoder_out = torch.stack(out_list, dim=0)

            return decoder_out

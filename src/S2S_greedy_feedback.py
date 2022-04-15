import math
import random
from copy import deepcopy

import torch
from torch import nn

USE_CUDA = torch.cuda.is_available()
MAX_OUTPUT_LENGTH = 46


class Beam:  # the class save the beam node
    def __init__(self, score, input_var, hidden, feed_hidden, all_output):
        self.score = score
        self.input_var = input_var
        self.hidden = hidden
        self.feed_hidden = feed_hidden
        self.all_output = all_output


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, input_length):
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)

        pack = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_length)
        outputs, hidden, = self.rnn(pack)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        problem_output = output[-1, :, :self.hid_dim] + output[0, :, self.hid_dim:]
        output = output[:, :, self.hid_dim:] + output[:, :, :self.hid_dim]
        return output, problem_output


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs, seq_mask=None, dropout=None):
        hidden = hidden.repeat(encoder_outputs.size(0), 1, 1)  # S x B x H
        # For each position of encoder outputs
        this_batch_size = encoder_outputs.size(1)
        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, 2 * self.hidden_size)
        attn_energies = self.score(torch.tanh(self.attn(energy_in)))  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(encoder_outputs.size(0), this_batch_size).transpose(0, 1)  # B x S
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
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill_(num_mask, -1e12)
        return score


class MinusGate(nn.Linear):
    def __init__(self, hidden_size, stuff_num, n_layers):
        super(MinusGate, self).__init__(hidden_size, stuff_num, n_layers)
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.stuff_num = stuff_num
        self.i_gates = nn.Linear(hidden_size * stuff_num, hidden_size)
        self.c_gates = []
        self.z_gates = []
        self.norm = []
        for _ in range(n_layers):
            self.c_gates.append(nn.Linear(hidden_size * stuff_num, hidden_size))
            self.z_gates.append(nn.Linear(hidden_size * 3, hidden_size))
            self.norm.append(nn.LayerNorm(hidden_size))
        self.c_gates = nn.ModuleList(self.c_gates)
        self.z_gates = nn.ModuleList(self.z_gates)
        self.norm = nn.ModuleList(self.norm)
        self.gelu = torch.nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.out = nn.LayerNorm(hidden_size)

    def forward(self, input):
        finput = torch.relu(self.i_gates(torch.cat(input, -1)))
        for layer in range(self.n_layers):
            temp = self.norm[layer](finput)
            c = torch.relu(self.c_gates[layer](torch.cat(input, -1)))
            z = torch.relu(self.z_gates[layer](torch.cat([c, temp, input[0]], -1)))
            finput = finput - c * z
        return self.out(finput)


class UpdateGate(nn.Module):
    def __init__(self, hidden_size):
        super(UpdateGate, self).__init__()
        self.hidden_size = hidden_size

        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        # For each position of encoder outputs
        this_batch_size = encoder_outputs.size(1)

        energy_in = torch.cat((hidden, encoder_outputs), 2)

        attn_energies = self.score(torch.relu(self.attn(energy_in)))  # (S x B) x 1

        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return torch.sigmoid(attn_energies)

class Decoder(nn.Module):
    def __init__(self, hidden_size, n_layers=2, dropout=0.5):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.dropout = nn.Dropout(0.5)

        self.gf = MinusGate(hidden_size, 2, n_layers)
        self.gr = MinusGate(hidden_size, 3, 0)
        self.gs = MinusGate(hidden_size, 2, n_layers)
        self.gc = MinusGate(hidden_size, 2, n_layers)

        self.attn = Attn(hidden_size)
        self.fed = UpdateGate(hidden_size)
        self.sel = Attn(hidden_size)

        print('decoder layer: ', n_layers)

    def forward(self, input_ids, feed_hidden, output, word_dict_vec, encoder_outputs, seq_mask):
        trg_one_hot = nn.functional.one_hot(input_ids, num_classes=word_dict_vec.size(1)).float()
        last_embedded = trg_one_hot.unsqueeze(1).bmm(word_dict_vec).transpose(0, 1)

        solution = self.gr([feed_hidden[:1], output.unsqueeze(0), last_embedded])

        temp = solution.repeat(feed_hidden.size(0), 1, 1)
        feedback = self.gf([feed_hidden, temp])
        gate = self.fed(solution, feed_hidden)
        feed_hidden = (1 - gate) * feed_hidden + gate * feedback

        temp = solution.repeat(feed_hidden.size(0), 1, 1)
        values = self.gs([temp, feed_hidden])
        attn_weights1 = self.sel(solution, feed_hidden)
        new_hidden = attn_weights1.bmm(values.transpose(0, 1)).transpose(0, 1)

        attn_weights2 = self.attn(new_hidden, encoder_outputs, seq_mask=seq_mask)
        context = attn_weights2.bmm(encoder_outputs.transpose(0, 1)).transpose(0, 1)
        output = self.gc([new_hidden, context])

        return output.squeeze(0), torch.cat((feed_hidden, new_hidden), dim=0)


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
        nn.init.normal_(self.embedding_weight, mean=0.0, std=0.5 / math.sqrt(hidden_size))
        self.init = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

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
        feed_hidden = problem.unsqueeze(0)
        output = self.norm(torch.relu(self.init(problem)))
        input = torch.LongTensor([output_lang.word2index["SOS"]] * batch_size)
        if USE_CUDA:
            input = input.cuda()
        if teacher_forcing > random.random():
            for t in range(0, max_len - 1):
                output, feed_hidden = self.decoder.forward(input, feed_hidden, output,
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
                output, feed_hidden = self.decoder.forward(input, feed_hidden, output,
                                                           word_dict_vec, encoder_out, seq_mask)

                score_list = self.score(output.unsqueeze(1), word_dict_vec, num_mask)

                out_list.append(score_list)

                input = torch.argmax(score_list, dim=-1)

            decoder_out = torch.stack(out_list, dim=0)

            return decoder_out

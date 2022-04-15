import math
import pickle
import time

import torch
from sklearn.utils import shuffle
from tqdm import tqdm

import torch.nn

from src.gsfsf import *
from src.pre_data import *
from src.train_and_evaluate import *
import warnings

val_part = "test"
model_name = "gsfsf"
warnings.filterwarnings("ignore")
USE_CUDA = torch.cuda.is_available()
path = "models/electra2seq"

PAD_token = 0

transformer_name = "hfl/chinese-electra-180g-base-discriminator"
dropout = 0.5
embedding_size = 128
hidden_size = 768
n_layer = 4
batch_size = 32

data = load_raw_data("data/math23k_train.json")
pairs, generate_nums, copy_nums = transfer_num(data)
temp_pairs = []
for p in pairs:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs_trained = temp_pairs

data = load_raw_data("data/math23k_test.json")
pairs, _, _ = transfer_num(data)
temp_pairs = []
for p in pairs:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs_tested = temp_pairs

if val_part == "train":
    pairs = pairs_trained
else:
    pairs = pairs_tested

input_lang = Lang()
output_lang = Lang()

with open(path + '/math_input_lang.pkl', 'rb') as f:
    input_lang = pickle.load(f)  # read file and build object

with open(path + '/math_output_lang.pkl', 'rb') as f:
    output_lang = pickle.load(f)  # read file and build object

stuff_size = output_lang.num_start + len(generate_nums) + 4
print(output_lang.index2word, " stuff size:", stuff_size)
pairs = prepare_test(pairs, input_lang, output_lang, tree=True)

input_batches_test, input_lengths_test, output_batches_test, output_lengths_test, nums_batches_test, num_stack_batches_test, num_pos_batches_test, _ = prepare_test_batch(
    pairs, 1)

# Initialize models
enc = Encoder(input_dim=input_lang.n_words, emb_dim=hidden_size, hid_dim=hidden_size, n_layers=n_layer,
              model_name=transformer_name, dropout=dropout)
dec = Decoder(hidden_size=hidden_size, n_layers=n_layer, dropout=dropout)

model = Seq2Seq(encoder=enc, decoder=dec, stuff_size=stuff_size, hidden_size=hidden_size,
                output_lang=output_lang)
# the embedding layer is  only for generated number embeddings, operators, and paddings

# Move models to GPU


print('loading weights...')
model.load_state_dict(
    torch.load(path + "/" + model_name + "_math_" + str(n_layer) + "layer.bin", map_location=torch.device('cpu')))
if USE_CUDA:
    model.cuda()

generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])

value_ac = 0
equation_ac = 0
eval_total = 0
id = 1
all = []

with torch.no_grad():
    for idx in range(len((input_lengths_test))):

        out = evaluate_with_beam_search(input_batches_test[idx], input_lengths_test[idx], output_batches_test[idx],
                                 output_lang, num_pos_batches_test[idx], model)



        val_ac, equ_ac, test, tar = compute_prefix_tree_result(out, output_batches_test[idx][0],
                                                               output_lang, nums_batches_test[idx][0],
                                                               num_stack_batches_test[idx][0])

        s = []
        for j in input_batches_test[idx][0]:
            s.append(input_lang.index2word[j])

        if val_ac:
            value_ac += 1
        if equ_ac:
            equation_ac += 1
        eval_total += 1


        def tostr(s):
            t = ''
            for w in s:
                t += w + ' '
            return t


        temp = {}
        temp['id'] = id
        temp['equ'] = equ_ac
        temp['val'] = val_ac
        temp['tar'] = tar
        temp['test'] = test
        temp['text'] = s
        # temp['gate'] = gate[i].tolist()
        # temp['attn'] = attn[i].tolist()

        all.append(temp)
        print('******************************************')
        print(id)
        print(tostr(s))

        print(equ_ac, val_ac)
        print(tar)
        print(test)
        print(temp)
        # print(temp['gate'])
        # print(temp['attn'])
        print(equation_ac, value_ac, eval_total, float(value_ac) / eval_total)
        print('******************************************')

        id += 1

# name = model_name + "_math_" + val_part + "_" + str(n_layer) + "layer_with_attn.json"
# print(name)
# with open(name, 'w', encoding='utf-8') as f_obj:
#     json.dump(all, f_obj)

print(equation_ac, value_ac, eval_total)
print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)

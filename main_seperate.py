import argparse
import os
import torch
from tqdm import tqdm
from language_quality import extract_good_candidates_by_LQ, get_LQ_scores
from utils import read_candidates, initialize_train_test_dataset, to_method_object, convert_to_contexts_responses
import numpy as np
import pickle
import random


random.seed(7503)

# module 1  read the generated candidates----------------------------------------------------
dataset = "conan"
data_file = "all_candidates.txt"
#
# dir_path = os.path.dirname(os.path.realpath(__file__))
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('Start Main...')
# print('start module 1')
# ''' Read Candidates from Module 1 '''
candidates = read_candidates('./data/' + data_file)  # Load generated candidates from Module 1.
train_x_text, train_y_text, test_x_text, test_y_text = initialize_train_test_dataset(dataset)
contexts_train, responses_train = convert_to_contexts_responses(train_x_text, train_y_text)
# module 1  read the generated candidates----------------------------------------------------



# module 2  Candidates pruning by grammaticality----------------------------------------------------

LQ_thres=0.52
model_name = 'bert-base-cased'
saved_pretrained_CoLA_model_dir = './tmp/grammar_cola'
to_test_candidates = candidates[:]
LQ_scores = get_LQ_scores(to_test_candidates, model_name, saved_pretrained_CoLA_model_dir)
avg_LQ = np.mean(LQ_scores)
#
print("average LQ before pruning ", avg_LQ )
#
scores = {i: j for i, j in zip(to_test_candidates, LQ_scores) if j > LQ_thres}
good_candidates = list(scores.keys())
good_candidates = list(set(good_candidates))
sum = 0
count = 0
for key in scores:
    sum += scores[key]
    count+=1
avg_LQ_pruned = sum / count
print("average LQ after pruning ",avg_LQ_pruned  )
#
#
# contexts_train_cp = contexts_train[:]
# responses_train_cp = responses_train[:]

method = to_method_object('TF_IDF')
method.train(contexts_train, responses_train)
good_candidates_index = method.sort_responses(test_x_text, candidates, min(100, len(candidates)))
good_candidates = [[candidates[y] for y in x] for x in good_candidates_index]
idx = "_11_18_all"
lq_score = [avg_LQ, avg_LQ_pruned]
#

# with open('test_x_text'+idx, 'wb') as f:
#     pickle.dump(test_x_text,f)

with open('store/context_train_redit' + idx, 'wb') as f:
    pickle.dump(contexts_train, f)
with open('store/responses_train_redit' +idx, 'wb') as f:
    pickle.dump(responses_train, f)
with open('store/scores' +idx , 'wb') as f:
    pickle.dump(scores, f)
with open('store/good_candidate' +idx, 'wb') as f:
    pickle.dump(good_candidates, f)
with open('store/lq_score' + idx , 'wb') as f:
    pickle.dump(lq_score, f)

# module 2  Candidates pruning by grammaticality----------------------------------------------------

# idx = "_11_17"

# with open('store/context_train_redit' + idx, 'rb') as f:
#     context_train = pickle.load(f)
# with open('store/responses_train_redit' +idx, 'rb') as f:
#     responses_train = pickle.load(f)
# with open('store/scores' +idx , 'rb') as f:
#     scores = pickle.load(f)
# with open('store/good_candidate' +idx, 'rb') as f:
#     good_candidates = pickle.load(f)
# with open('store/lq_score' + idx , 'rb') as f:
#     lq_score = pickle.load(f)

# module 3   Response Selection----------------------------------------------------

# METHODS = ['TF_IDF', 'BM25', 'USE_SIM', 'USE_MAP', 'USE_LARGE_SIM', 'USE_LARGE_MAP', 'ELMO_SIM', 'ELMO_MAP',
#            'BERT_SMALL_SIM', 'BERT_SMALL_MAP', 'BERT_LARGE_SIM', 'BERT_LARGE_MAP', 'USE_QA_SIM', 'USE_QA_MAP',
#            'CONVERT_SIM', 'CONVERT_MAP']
#


#
# method_name = 'BERT_SMALL_MAP'
# print(method_name)
# method = to_method_object(method_name)
# method.train(context_train, responses_train)
#
# output = []
# for i, test_i in enumerate(tqdm(test_x_text)):
#     predictions = method.rank_responses([test_i], good_candidates[i])
#     output.append(good_candidates[i][predictions.item()])
#
# with open('output'+idx, 'wb') as f:
#     pickle.dump(output,f)
# with open('test_x_text'+idx, 'wb') as f:
#     pickle.dump(test_x_text,f)
#
# with open('output'+idx + '.txt', 'w', encoding='utf-8') as f:
#     for line in output:
#         f.write(line + '\n')
#
# with open('test_x_text' + idx + '.txt' , 'w',encoding='utf-8') as f:
#     for line in test_x_text:
#         f.write(line + '\n')


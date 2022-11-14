import csv
import numpy as np
import pandas
import pickle
data_path = "data/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech-master/reddit.csv"
# data_path = "data/reddit_for_VAE.txt"
# with open(data_path, encoding='utf8') as f:
#     lines = f.readlines()


# from numpy import genfromtxt
# x = np.loadtxt(data_path,encoding='utf-8')
# my_data = genfromtxt(data_path,encoding='utf-8')
#
lines= []
# names = ['0','1','2', '3' ]
# data = pandas.read_csv(data_path, names=names)

# def split_counter_speechs(counter_speechs):

def processce_sent(s,out):
    s_id = 0
    while s_id < len(s):
        sent = s[s_id]
        if len(sent) < 3:
            s_id += 1
        else:
            whole = ""
            if sent[1] in ['\'', '\"' ]  and sent[-1] not in ['\'', '\"' ]:
                sent = sent[2:]
                whole += sent
                s_id += 1
                print("start ", s_id,sent)
                while s_id < len(s) and ( not (s[s_id][-1] in ['\'', '\"' ])):
                    print(s_id)
                    print(s[s_id])
                    whole += s[s_id]
                    s_id += 1
                print("end", s_id)


                whole += s[s_id][:-1]
                # whole.strip("\"")
                out.append(whole)
            elif sent[1] in ['\'', '\"' ]  and sent[-1] in ['\'', '\"' ]:

                sent = sent[2:-1]
                out.append(sent)
            elif sent[0] == "\"":
                sent = sent.strip("\"")
                sent = sent.strip()
                out.append(sent)
            elif sent[1] == "\"":
                sent = sent.strip("\"")
                sent = sent.strip()
                out.append(sent)
            s_id += 1



with open(data_path,'r', encoding='utf-8') as f:
    csvreader = csv.reader(f)
    # for col in csvreader:
    #     print(col)
    for row in csvreader:
        lines.append(row)
        # idx = row[0]
        # print(row)

out = []
for row in lines:
    s = row[3]
    s = s.strip(']')
    s = s.strip('[')
    s = s.split(",")
    processce_sent(s, out)



with open('reddit_for_VAE.txt', 'w', encoding='utf-8') as f:
    for line in out:
        f.write(line + "\n")
# s = row[3].strip('[')
# s = s.strip(']')
# s = s.split("\"")
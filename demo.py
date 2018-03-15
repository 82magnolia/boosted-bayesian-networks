import numpy as np
import nltk
from nltk import word_tokenize
import re
import pandas as pd


f = open("amazon_cells_labelled.txt")
f2 = open("imdb_labelled.txt")
f3 = open("yelp_labelled_fix_test.txt")
f4 = open("train_data.txt")

str_list = ["" for i in range(1000)]
for line,i in zip(f,range(1000)):
    str_list[i] = re.sub('[.,]','',line)

str_list2 = ["" for i in range(1000)]
for line,i in zip(f2,range(1000)):
    str_list2[i] = re.sub('[.,]','',line)

str_list3 = ["" for i in range(11000)]
for line,i in zip(f3,range(11000)):
    str_list3[i] = re.sub('[.,]','',line)

length = 575
str_list4 = ["" for i in range(length)]
for line,i in zip(f4,range(length)):
    str_list4[i] = re.sub('[.,]','',line)

str_super = str_list + str_list2 + str_list3 + str_list4

for i in range(len(str_super)):
    str_super[i] = str_super[i].strip('\t0\n')
    str_super[i] = str_super[i].strip('\t1\n')
    str_super[i] = str_super[i].lower()

token_list = [[] for i in range(1000)]
for i in range(1000):
    token_list[i] = word_tokenize(str_list[i])

token_list2 = [[] for i in range(1000)]
for i in range(1000):
    token_list2[i] = word_tokenize(str_list2[i])

token_list3 = [[] for i in range(11000)]
for i in range(11000):
    token_list3[i] = word_tokenize(str_list3[i])

token_list4 = [[] for i in range(length)]
for i in range(length):
    token_list4[i] = word_tokenize(str_list4[i])

label = [token_list[i][len(token_list[i]) - 1] for i in range(1000)]

label2 = [token_list2[i][len(token_list2[i]) - 1] for i in range(1000)]

label3 = [token_list3[i][len(token_list3[i]) - 1] for i in range(11000)]

label4 = [token_list4[i][- 1] for i in range(length)]

super_label = label + label2 + label3 + label4

super_label = ['1' if super_label[i] == '1' else '-1' for i in range(len(super_label))]

str_data = [token_list[i][:len(token_list[i]) - 1] for i in range(1000)]

str_data = [[str_data[i][j].lower() for j in range(len(str_data[i]))] for i in range(1000)]

str_data2 = [token_list2[i][:len(token_list2[i]) - 1] for i in range(1000)]

str_data2 = [[str_data2[i][j].lower() for j in range(len(str_data2[i]))] for i in range(1000)]

str_data3 = [token_list3[i][:len(token_list3[i]) - 1] for i in range(11000)]

str_data3 = [[str_data3[i][j].lower() for j in range(len(str_data3[i]))] for i in range(11000)]

str_data4 = [token_list4[i][:len(token_list4[i]) - 1] for i in range(length)]

str_data4 = [[str_data4[i][j].lower() for j in range(len(str_data4[i]))] for i in range(length)]

str_super_data = str_data + str_data2 + str_data3 + str_data4

from collections import Counter

super_positive = [str_super_data[i] if super_label[i] == '1' else None for i in range(len(str_super_data))]
super_negative = [str_super_data[i] if super_label[i] == '-1' else None for i in range(len(str_super_data))]

super_pos = Counter([])
for i in range(len(super_positive)):
    super_pos += Counter(super_positive[i])

super_neg = Counter([])
for i in range(len(super_negative)):
    super_neg += Counter(super_negative[i])

super_pos = dict(super_pos)
super_neg = dict(super_neg)

pos_vocab = list(super_pos.keys())
neg_vocab = list(super_neg.keys())

num_vocab = 35000

def baseline(inptstr, phi_pos, phi_neg, phi_ypos, phi_yneg, num_vocab, pos_voc, neg_voc):
    tok_list = word_tokenize(inptstr)
    log_prob_pos = 0
    log_prob_neg = 0
    for word in tok_list:
        if(word in pos_voc):
            log_prob_pos += np.log(phi_pos[word])
        else:
            log_prob_pos += np.log(1/num_vocab)
    for word in tok_list:
        if(word in neg_voc):
            log_prob_neg += np.log(phi_neg[word])
        else:
            log_prob_neg += np.log(1/num_vocab)
    log_prob_pos += np.log(phi_ypos)
    log_prob_neg += np.log(phi_yneg)
    if(log_prob_pos >= log_prob_neg):
        return '1'
    else:
        return '-1'

def weight_train(s_data, l_data, pos_voc, neg_voc, weight):
    data_size = len(s_data)
    tot_weight = sum([weight[i] for i in range(data_size)])
    weight_pos = sum([weight[i] if l_data[i] == '1' else 0 for i in range(data_size)])
    weight_neg = sum([weight[i] if l_data[i] == '-1' else 0 for i in range(data_size)])
    phi_ypos = weight_pos/tot_weight
    phi_yneg = weight_neg/tot_weight
    phi_pos_num = {i:0 for i in pos_voc}
    phi_neg_num = {i:0 for i in neg_voc}
    for i in range(data_size):
        str_size = len(s_data[i])
        if(l_data[i] == '1'):
            for j in range(str_size):
                phi_pos_num[s_data[i][j]] += weight[i]
        else:
            for j in range(str_size):
                phi_neg_num[s_data[i][j]] += weight[i]
    phi_pos = {i:phi_pos_num[i]/weight_pos for i in pos_voc}
    phi_neg = {i:phi_neg_num[i]/weight_neg for i in neg_voc}
    return phi_pos, phi_neg, phi_ypos, phi_yneg

def error(s_raw, s_data, l_data, phi_pos, phi_neg, phi_ypos, phi_yneg, weight, num_vocab, pos_voc, neg_voc):
    prediction = predict(s_raw, s_data, phi_pos, phi_neg, phi_ypos, phi_yneg, num_vocab, pos_voc, neg_voc)
    data_size = len(s_data)
    weighted_error = sum([weight[i] if prediction[i] != l_data[i] else 0 for i in range(data_size)])
    return weighted_error,prediction

def predict(s_raw, s_data, phi_pos, phi_neg, phi_ypos, phi_yneg, num_vocab, pos_voc, neg_voc):
    data_size = len(s_data)
    prediction = [baseline(s_raw[i], phi_pos, phi_neg, phi_ypos, phi_yneg, num_vocab, pos_voc, neg_voc) for i in range(data_size)]
    return prediction

def boosting(s_raw, s_data, l_data, pos_voc, neg_voc, steps, num_vocab):
    weight = [1/len(s_data) for i in range(len(s_data))]
    beta = [0 for i in range(steps)]
    phi_pos = [{} for i in range(steps)]
    phi_neg = [{} for i in range(steps)]
    phi_ypos = [0 for i in range(steps)]
    phi_yneg = [0 for i in range(steps)]
    for i in range(steps):
        print('step',i)
        phi_pos[i], phi_neg[i], phi_ypos[i], phi_yneg[i] = weight_train(s_data, l_data, pos_voc, neg_voc, weight)
        weighted_error,prediction = error(s_raw, s_data,l_data,phi_pos[i],phi_neg[i],phi_ypos[i],phi_yneg[i], weight, num_vocab,pos_voc, neg_voc)
        beta[i] = 0.5*np.log((1 - weighted_error)/weighted_error)
        for j in range(len(s_data)):
            weight[j] = weight[j]*np.exp(-beta[i]*float(l_data[j])*float(prediction[j]))
    return beta, phi_pos, phi_neg, phi_ypos, phi_yneg

def boost_predict(inptstr, beta, phi_pos, phi_neg, phi_ypos, phi_yneg, num_vocab, pos_voc, neg_voc):
    pred = 0
    for i in range(len(beta)):
        pred += beta[i]*float(baseline(inptstr, phi_pos[i], phi_neg[i], phi_ypos[i], phi_yneg[i], num_vocab, pos_voc, neg_voc))
    if(pred >= 0):
        return '1'
    else:
        return '-1'

def threshold(value): ##return true by probability of value
    rnd = np.random.rand()
    if rnd < value:
        return True
    else:
        return False

def gibbs_sample(phi_pos, phi_neg, phi_ypos, phi_yneg, pos_voc, neg_voc, num_sample):
    for smp in range(num_sample):
        if(threshold(phi_ypos)):
            print("positive:")
            for i in range(len(pos_voc)):
                if(threshold(phi_pos[pos_voc[i]])):
                    print(pos_voc[i],end = ' ')
            print()
        else:
            print("negative:")
            for i in range(len(neg_voc)):
                if(threshold(phi_neg[neg_voc[i]])):
                    print(neg_voc[i],end = ' ')
            print()


beta, phi_pos, phi_neg, phi_ypos, phi_yneg = boosting(str_super, str_super_data, super_label, pos_vocab, neg_vocab,5, num_vocab)

while(True):
    print("enter sentence:")
    inptstr = input()
    if(inptstr == 'end'):
        break
    if(inptstr == 'Gibbs'):
        gibbs_sample(phi_pos[0],phi_neg[0],phi_ypos[0],phi_yneg[0],pos_vocab, neg_vocab, 100)
    else:
        print(boost_predict(inptstr, beta, phi_pos, phi_neg, phi_ypos, phi_yneg, num_vocab, pos_vocab, neg_vocab))



#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/5 9:13
# @Author  : Wang Jixin
# @File    : prepar_curpus.py
# @Software: PyCharm
import os.path

import pandas as pd
import jieba
import json
import os
import re

def generate_words_dict(word_list:list[list]):
    temp_words_list = []
    for words in words_list:
        temp_words_list += words
    temp_words_set = set(temp_words_list)
    tag2words = {tag:word for tag,word in enumerate(temp_words_set)}
    words2tag = {word:tag for tag, word in enumerate(temp_words_set)}
    if not os.path.exists('../output/tag2word.json'):
        with open('../output/tag2word.json','w',encoding='utf-8') as f:
            json.dump(tag2words,f,ensure_ascii=False)
    if not os.path.exists('../output/word2tag.json'):
        with open('../output/word2tag.json', 'w', encoding='utf-8') as f:
            json.dump(words2tag, f,ensure_ascii=False)
    print('words dict finished!')



def remove_symbols(words:str):
    symbols = [',','.','!','?','！','，','。','？']
    for symbol in symbols:
        words = words.replace(symbol,"")
    return words

def sen_cut(sen_list:list):
    words_list = []
    for sen in sen_list:
        sen = remove_symbols(sen)
        words_list.append(jieba.lcut(sen))
    return words_list


def generate_sen_list(data:pd.DataFrame):
    sen_list = []
    for line in data.itertuples():
        sen_list.append(line.review)
    return sen_list


def generate_curpus(cut_train_word_df:pd.DataFrame):
    cols = cut_train_word_df.columns
    curpus = []
    for tup in cut_train_word_df.itertuples():

        if len(tup) == 3:
            curpus += tup.cols[0]
    curpus_set = set(curpus)





if __name__ == '__main__':
    cut_train_word_df = pd.read_csv('../datasets/cut_train_word_df.csv',index_col=False)
    generate_curpus(cut_train_word_df)

    # data = pd.read_csv('../datasets/waimai_10k.csv',index_col=False)
    # sen_list = generate_sen_list(data)
    # words_list = sen_cut(sen_list)
    # generate_words_dict(words_list)
    # print(words_list)
    #
    #
    # test = '吃饭了吗?'
    # y = remove_symbols(test)
    # print(y)

    # print(sen_list)

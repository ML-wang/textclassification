#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/5 21:46
# @Author  : Wang Jixin
# @File    : tokenier.py
# @Software: PyCharm
import os

import pandas as pd
import jieba
from config.config import Config

import json
def generate_words_dict(config:Config):
    cut_train_word_df = pd.read_csv(config.train_data_path, index_col=False)
    cut_train_word_df['review'] = cut_train_word_df['review'].apply(remove_symbols).apply(lambda x: jieba.lcut(x))
    temp_word_list = []
    if not os.path.exists(config.cut_train_word_path):
        cut_train_word_df.to_csv(config.cut_train_word_path, index=False)
    for word in cut_train_word_df['review'].items():
        temp_word_list += word[1]
    words_set = set(temp_word_list)
    tag2words = {tag: word for tag, word in enumerate(words_set,2)}
    words2tag = {word: tag for tag, word in enumerate(words_set,2)}
    tag2words['0'] = 'UNK'
    tag2words['1'] = 'PAD'
    words2tag['UNK'] = 0
    words2tag['PAD'] = 1
    if not os.path.exists(config.tag2word_path):
        with open(config.tag2word_path,'w',encoding='utf-8') as f:
            json.dump(tag2words,f,ensure_ascii=False)
    if not os.path.exists(config.word2tag_path):
        with open(config.word2tag_path, 'w', encoding='utf-8') as f:
            json.dump(words2tag, f,ensure_ascii=False)
    print('words dict finished!')

    return tag2words, words2tag


def remove_symbols(words: str):
    symbols = [',', '.', '!', '?', '！', '，', '。', '？']
    for symbol in symbols:
        words = words.replace(symbol, "")
    return words


def tokenier(config:Config):
    with open(config.word2tag_path,encoding='utf-8') as f:
        data = f.readlines()[0]
        words2tag = json.loads(data)
    # print(words2tag)
    train_words_df = pd.read_csv(config.train_data_path)
    words_list = []
    labels = []
    for sen in train_words_df.itertuples():
        words_list.append(jieba.lcut(sen.review))
        labels.append(sen.label)
    token_words_list = []
    mask = []
    for word_list in words_list:
        temp_list = []
        if len(word_list) <= config.max_len:
            mask.append(len(word_list)*[1]+(20-len(word_list))*[0])
            word_list.extend(['PAD']*(20-len(word_list)))
        else:
            word_list = word_list[:20]
            mask.append(20* [1])
        for word in word_list:
            temp_list.append(words2tag.get(word,0))
        token_words_list.append(temp_list)
    assert len(labels) == len(token_words_list)
    return token_words_list,mask,labels




if __name__ == '__main__':
    config = Config()
    tag2words, words2tag = generate_words_dict(config)
    token_words_list, mask, labels =  tokenier(config)
    print('')





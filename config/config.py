#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/5 9:11
# @Author  : Wang Jixin
# @File    : config.py
# @Software: PyCharm

import os
import json


CURRENT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
with open('../output/tag2word.json',encoding='UTF-8') as f:
    data = f.readlines()[0]
    tag2word = json.loads(data)

class Config:
    def __init__(self):
        self.max_len = 20
        self.vocab = len(tag2word)
        self.train_data_path = os.path.join(CURRENT_DIR_PATH, '../datasets/train.csv')
        self.val_data_path = os.path.join(CURRENT_DIR_PATH, '../datasets/val.csv')
        self.test_data_path = os.path.join(CURRENT_DIR_PATH, '../datasets/test.csv')
        self.tag2word_path = os.path.join(CURRENT_DIR_PATH,'../output/tag2word.json')
        self.word2tag_path = os.path.join(CURRENT_DIR_PATH,'../output/word2tag.json')
        self.cut_train_word_path = os.path.join(CURRENT_DIR_PATH,'../datasets/cut_train_word_df.csv')
        self.bert = False
        self.batch_size  = 8




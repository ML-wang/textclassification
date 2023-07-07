#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/6 16:55
# @Author  : Wang Jixin
# @File    : model_config.py
# @Software: PyCharm


import json


with open('../output/word2tag.json',encoding='UTF-8') as f:
    data = f.readlines()[0]
    json_data = json.loads(data)
print(len(json_data))

class ModelConfig:
    def __init__(self):
        self.epochs = 50
        self.batch_sizes = 8
        self.lr = 0.001
        self.embed_size = 100
        self.hidden_size = 256
        self.num_layers = 2
        self.dropout = 0.1
        self.vocab = len(json_data)
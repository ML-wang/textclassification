#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/6 16:55
# @Author  : Wang Jixin
# @File    : model_config.py
# @Software: PyCharm


import json
import torch

from datetime import datetime
import os
from config.config import Config

MODEL_TIME = datetime.now().strftime('%y-%m-%d-%H_%M_%S')
CURRENT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))


con = Config()
with open('../output/tag2word.json',encoding='UTF-8') as f:
    data = f.readlines()[0]
    json_data = json.loads(data)
print(len(json_data))

class Model_Config:
    def __init__(self):
        self.model_name = 'fast_text'
        self.vocab = len(json_data)
        self.epochs = 50
        self.batch_sizes = 8
        self.lr = 0.001
        self.embed_size = 100
        self.hidden_size = 786 if self.model_name == 'bert' else 100
        self.num_layers = 2
        self.dropout = 0.1
        self.vocab = len(json_data)
        self.n_classes = 2
        self.save_path = os.path.join(CURRENT_DIR_PATH,f'../save_models/{MODEL_TIME}_{self.model_name}','.ckpt') # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.bert_path = '../bert-base-chinese'
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练









if __name__ == '__main__':
    config = Bert_Config()
    print('')

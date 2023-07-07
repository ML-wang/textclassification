#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/5 9:13
# @Author  : Wang Jixin
# @File    : Bert.py
# @Software: PyCharm


from transformers import BertModel
from utils.tokenier import tokenier
from config.model_config import Bert_Config
import torch.nn as nn
# from utils.dataloader import mydataloader
config = Bert_Config()
config.bert = True



class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out


if __name__ == '__main__':
    bert_model = Model(config)
    print(bert_model)
    for token, labels in mydataloader:
        print(token)
        print(labels)
        break



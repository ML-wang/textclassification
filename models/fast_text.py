#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/5 9:13
# @Author  : Wang Jixin
# @File    : fast_text.py
# @Software: PyCharm

import torch
import torch.nn as nn
from config import Model_Config


class Fast_Text(nn.Module):
    def __init__(self, model_config:Model_Config):
        super(Fast_Text, self).__init__()
        #创建embedding
        self.embed = nn.Embedding(model_config.vocab, model_config.embed_size)  #embedding初始化，需要两个参数，词典大小、词向量维度大小
        self.embed.weight.requires_grad = True #需要计算梯度，即embedding层需要被训练
        self.fc = nn.Sequential(              #序列函数
            nn.Linear(model_config.embed_size, model_config.hidden_size),  #这里的意思是先经过一个线性转换层
            nn.BatchNorm1d(model_config.hidden_size),      #再进入一个BatchNorm1d
            nn.ReLU(inplace=True),            #再经过Relu激活函数
            nn.Linear(model_config.hidden_size, model_config.n_classes)#最后再经过一个线性变换
        )

    def forward(self, x):
        x = self.embed(x)                     #先将词id转换为对应的词向量
        out = self.fc(torch.mean(x, dim=1))   #这使用torch.mean()将向量进行平均
        return out


if __name__ == '__main__':
    model_config = Model_Config()
    fast_text = Fast_Text(model_config)
    print(fast_text)



#
#
# class Fast_Text:
#     def __init__(self):

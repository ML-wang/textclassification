#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/5 9:13
# @Author  : Wang Jixin
# @File    : BiLstm.py
# @Software: PyCharm

import torch
import torch.nn as nn
from crf import CRF
import torch
from torchtext.vocab import GloVe
from config.model_config import ModelConfig


model_config = ModelConfig()

class BiLSTM(nn.Module):
    def __init__(self, model_config:ModelConfig):
        # num_layers=2表示有两个方向（双向），LSTM的参数bidirectional=True表示双向LSTM,默认为False（单向）
        super().__init__()
        self.embedding = nn.Embedding(model_config.vocab, model_config.embed_size)
        self.rnn = nn.LSTM(model_config.embed_size,
                           model_config.hidden_size,
                           num_layers=model_config.num_layers,
                           bidirectional=True, dropout=model_config.dropout)
        self.fc = nn.Linear(2 * model_config.hidden_size, 2)

    # xavier初始化参数
        self._reset_parameters()


    # 使用预训练的词向量，将其冻结，训练期间不再更新
        glove = GloVe(name="42B", dim=300)
        self.embedding = nn.Embedding.from_pretrained(glove.get_vecs_by_tokens(model_config.vocab.get_itos()),
                                                  freeze=True)

    def forward(self, x):
        x = self.embedding(x).transpose(0, 1)
        """
        LSTM的输出有三部分:
        第一部分:output,LSTM所有时间步的隐层结果
        output:[batch_size,seq_len,hidden_dim * X] 双向时X=2,单向时X=1
        第二部分:h为最后一个时间步的隐层结果
        第三部分:c为最后一个时间步的cell状态
        (h, c) = [num_layers * X, batch_size, hidden_dim]
        """
        _, (h_n, _) = self.rnn(x)
        output = self.fc(torch.cat((h_n[-1], h_n[-2]), dim=-1))
        return output

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


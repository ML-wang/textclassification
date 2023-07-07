#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/5 9:11
# @Author  : Wang Jixin
# @File    : train.py
# @Software: PyCharm

from config.model_config import Model_Config
import torch.optim as optim
import torch.nn as nn
from transformers import BertModel
from utils.dataloader import MyDatasets
from torch.utils.data import DataLoader
from models import MLP,BertModel,BiLSTM,Fast_Text


model_config = Model_Config()


def training(model_config:Model_Config,mydatasets:MyDatasets):
    if model_config.model_name == 'bert':
        model = BertModel.from_pretrained('bert-base-chinese')
    elif model_config.model_name == 'mlp':
        model = MLP()
    elif model_config.model_name == 'bilstm':
        model = BiLSTM()
    elif model_config.model_name == 'fast_text':
        model = Fast_Text()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    epochs_loss = []
    data = MyDatasets()
    mydatasets = DataLoader(data)
    for epoch in model_config.epochs:
        for epoch,(token,labels) in enumerate(mydatasets):
            print(f'This is {epoch+1} epoch')
            output = model()





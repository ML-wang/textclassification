#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/5 9:11
# @Author  : Wang Jixin
# @File    : train.py
# @Software: PyCharm

from config.model_config import ModelConfig
from models.BiLstm import BiLSTM
import torch.optim as optim
import torch.nn as nn
from utils.dataloader import MyDatasets
from torch.utils.data import DataLoader


model_config = ModelConfig()


def training(model_config:ModelConfig,mydatasets:MyDatasets):
    model = BiLSTM(model_config)
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    epochs_loss = []
    data = MyDatasets()
    mydatasets = DataLoader(data)
    for epoch in model_config.epochs:
        for epoch,(token,labels) in enumerate(mydatasets):
            print(f'This is {epoch+1} epoch')
            output = model()





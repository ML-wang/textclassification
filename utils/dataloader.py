#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/6 16:58
# @Author  : Wang Jixin
# @File    : dataloader.py
# @Software: PyCharm



from torch.utils.data import DataLoader,Dataset
from tokenier import generate_words_dict,tokenier
from config.config import Config
import torch


class MyDatasets(Dataset):
    def __init__(self):
        token_words_list, mask, labels = tokenier(config)
        self.token_words_list = torch.tensor(token_words_list)
        self.mask = torch.tensor(mask)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.token_words_list[index],self.labels[index]








if __name__ == '__main__':
    config = Config()
    mydata = MyDatasets()
    mydataloader = DataLoader(mydata,batch_size=8)
    for token,labels in mydataloader:
        print(token)
        print(labels)
        break

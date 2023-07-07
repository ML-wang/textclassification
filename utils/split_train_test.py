#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/5 10:04
# @Author  : Wang Jixin
# @File    : split_train_test.py
# @Software: PyCharm
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split

def split_datasets(file_path):
    data = pd.read_csv(file_path,index_col=False)
    cols = data.columns
    train_x,test_val_x,train_y,test_val_y = train_test_split(data[[cols[1]]],data[[cols[0]]],
                                                     random_state=42,
                                                     test_size=0.3,
                                                     stratify=data[cols[0]])
    val_x,test_x,val_y,test_y = train_test_split(test_val_x[[cols[1]]],test_val_y[[cols[0]]],
                                                     random_state=42,
                                                     test_size=0.5,
                                                     stratify=test_val_y[cols[0]])
    if not os.path.exists('../datasets/train.csv'):
        train_df = pd.concat([train_x,train_y],axis=1)
        train_df.to_csv('../output/train.csv',index=False)
    if not os.path.exists('../datasets/val.csv'):
        val_df = pd.concat([val_x, val_y],axis=1)
        val_df.to_csv('../output/val.csv',index=False)
    if not os.path.exists('../datasets/test.csv'):
        test_df = pd.concat([test_x, test_y],axis=1)
        test_df.to_csv('../output/test.csv',index=False)
    return train_x,val_x,test_x,train_y,val_y,test_y











if __name__ == '__main__':
    # split_datasets('../datasets/waimai_10k.csv')
    train_x, val_x, test_x, train_y, val_y, test_y = split_datasets('../datasets/waimai_10k.csv')
    print(train_x)
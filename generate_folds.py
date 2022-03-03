# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 22:49:42 2022

@author: nawaz
"""
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

def gen_folds(dataset, label, number_of_folds):
        
    kf = KFold(n_splits = number_of_folds, shuffle = False)

    for train_index, test_index in kf.split(dataset):
        X_tr_va, X_test = dataset[train_index], dataset[test_index]
        y_tr_va, y_test = label[train_index], label[test_index]
        yield X_tr_va,y_tr_va,X_test,y_test
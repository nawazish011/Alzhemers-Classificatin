# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 22:56:09 2022

@author: nawaz
"""

import nibabel as nib
import numpy as np

from tensorflow.keras.optimizers import Adam
from load_data import get_dataset
from model import get_model
from generate_batches import gen_batches
from generate_folds import gen_folds


dataset_path = "E:/Alzheimer/numpy_dataset/"
# ds_path = 'E:/Alzheimer/testing_less_data_2'
n_folds = 5
batch_size = 16
n_epochs = 30
lr = 0.0001


def start_training():
    # dataset, labels = get_dataset(dataset_path)
    # print("Data loaded now start creating batch")
    # fold_gen = gen_folds(dataset,labels, n_folds)
    # gen_batch = gen_batches(dataset,labels,batch_size)
    
    model = get_model(summary=True)
    
    # count = 0;
    # for x_train,y_train,x_test,y_test in fold_gen:
    #     print(x_train.shape)
    #     print(x_test.shape)
    #     print(y_train.shape)
    #     print(y_test.shape)
    #     # break
    #     count = count+1     
    # print("Total count : ",count)
    
    
if __name__ == "__main__":
    start_training()
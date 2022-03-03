# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 22:39:55 2022

@author: nawaz
"""

from os.path import join
from os import listdir
import numpy as np

def get_dataset(ds_path):
    labels = list()
    dataset = list()
    classes_files = listdir(ds_path)
    for class_file in classes_files:
        class_name = class_file
        file_path = join(ds_path, class_name,class_name+'.npy')
        with open(file_path,'rb') as f:
            ds = np.load(f)
        
        print("data loading done") 
        
        for i in range(0, len(ds)):
            if class_name == 'AD':    
                labels.append([0,0,0,0,1])
            elif class_name == 'LMCI':    
                labels.append([0,0,0,1,0])
            elif class_name == 'MCI':    
                labels.append([0,0,1,0,0])
            elif class_name == 'EMCI':    
                labels.append([0,1,0,0,0])
            elif class_name == 'NC':    
                labels.append([1,0,0,0,0])
            dataset.append(ds[i])
        print("label loading done for class : ",class_name)
        
    ds_label = np.asarray(labels)
    dataset_final = np.asarray(dataset)
    
    dataset.clear()
    labels.clear()

    return dataset_final, ds_label
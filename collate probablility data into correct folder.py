# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:02:16 2022

@author: jwluc
"""

import os
from tqdm import tqdm
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm 
from sklearn import decomposition
import scipy
#import torch
#import cv2
#import pydicom
#import SimpleITK as sitk
import itertools
import random
from datetime import datetime
import sklearn.linear_model as lm



def resize_to_1D(data):
    temp = []
    for i in range(0,len(data)):
        for j in range(0,len(data[i])):
            temp.append(data[i][j])
    return temp

def delete_null(automatic, corrected, prob):
    #Find scans which have all class 0 in the automatic and corrected scans
    #Remove these from the automatic, corrected, and feature data arrays
    automatic_reduced = [] # np.array([], dtype = object)
    corrected_reduced = [] # np.array([], dtype = object)
    prob_reduced = []
    for i in range(0, len(automatic)):
        #temp_a = []
        #temp_c = []
        temp_auto = resize_to_1D(automatic[i])
        temp_corr = resize_to_1D(corrected[i])
        if (sum(temp_auto) + sum(temp_corr) != 0): 
            automatic_reduced.append(automatic[i])
            corrected_reduced.append(corrected[i])
            prob_reduced.append(prob[i])

    #automatic_reduced.append(temp_a)
    #corrected_reduced.append(temp_c)
           
    return automatic_reduced, corrected_reduced, prob_reduced



filepath_baseline = "\Data\Baseline scans"
filepath_week12 = "\Data\Week 12 scans"
scans=[1,3,6,7,10,12,13,19]
b = '\\baseline'
w = '\\week12'
filepath = '\\Probability data'
cwd = os.getcwd()
lstdir = os.listdir(cwd+filepath)
print(lstdir)
x = np.load(cwd+filepath+'\\'+lstdir[0]).astype(np.float32)
print(len(x),len(x[0]),len(x[0][0]))
number_of_slices = 113

r = ["",'_reduced']
for reduced in r:
    data_filepaths = []
    save_filepaths = []
    auto_filepaths = []
    corr_filepaths = []
    
    for scan in scans:
        temp_b = []
        temp_w = []
        for j in range(0,number_of_slices):
            temp_b.append(cwd+filepath+b+'Scan'+str(scan)+'_'+str(j)+'.npy')
        for j in range(0,number_of_slices):
            temp_w.append(cwd+filepath+w+'Scan'+str(scan)+'_'+str(j)+'.npy')  
        data_filepaths.append(temp_b)
        data_filepaths.append(temp_w)
        auto_filepaths.append(cwd+filepath_baseline+'\\scan'+str(scan)+'\\auto'+reduced+'\\')
        auto_filepaths.append(cwd+filepath_week12+'\\scan'+str(scan)+'\\auto'+reduced+'\\')    
        corr_filepaths.append(cwd+filepath_baseline+'\\scan'+str(scan)+'\\corrected'+reduced+'\\')
        corr_filepaths.append(cwd+filepath_week12+'\\scan'+str(scan)+'\\corrected'+reduced+'\\')    
        save_filepaths.append(cwd+filepath_baseline+'\\scan'+str(scan)+'\\prob'+reduced+'\\')
        save_filepaths.append(cwd+filepath_week12+'\\scan'+str(scan)+'\\prob'+reduced+'\\')    
    
    # Check folders exist and create them if not
    for path in save_filepaths:
        if not os.path.exists(path):
            os.makedirs(path)
            
    data = []
    auto = []
    corr = []
    #getting auto and corr data for calculating null images 
    for i in range(0,len(auto_filepaths)):
            arr = np.load(auto_filepaths[i]+'init.npy').astype(np.float32)
            auto.append(arr)
    for i in range(0,len(corr_filepaths)):
            arr = np.load(corr_filepaths[i]+'init.npy').astype(np.float32)
            corr.append(arr)
    #getting prob data
    for i in tqdm(range(0,len(data_filepaths))):
        temp =[]
        for j in range(0,len(data_filepaths[i])):
            arr = np.load(data_filepaths[i][j]).astype(np.float32)
            temp.append(arr)
        data.append(temp)
        #print(len(save_filepaths),len(data),len(data[i]),len(data[i][0]),len(data[i][0][0]))
        
    if reduced == '_reduced':
        prob = data
        data = []
        for i in tqdm(range(0,len(auto))):
            auto_red, corr_red, prob_red = delete_null(auto[i], corr[i], prob[i])
            data.append(prob_red)

        
    for i in tqdm(range(0,len(save_filepaths))): 
        np.save(save_filepaths[i] + 'init.npy', data[i],allow_pickle = True)
        print(save_filepaths[i],len(data),len(data[i]),len(data[i][0]),len(data[i][0][0]))

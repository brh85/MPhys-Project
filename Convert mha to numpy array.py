# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 12:47:40 2021

@author: Rory Harris
"""
import os
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk

data_filepaths = []
save_filepaths = []
filepath_old_baseline = "\Data_old\Baseline scans"
filepath_old_week12 = "\Data_old\Week 12 scans"
filepath_baseline = "\Data\Baseline scans"
filepath_week12 = "\Data\Week 12 scans"
name = [['FA14Saggital_TissueMap.mha','auto'],
        ['FA14Saggital_TissueMap_Corrected.mha','corrected']]
scans=[1,3,6,7,10,12,13,19]
data_possibilities = [0,1]
cwd = os.getcwd()

def merge_classes(data):
    # receives a 32x113(or z)x512x512 array
    for i in tqdm(range(0,len(data))):
        for j in range(0,len(data[i])):
            for k in range(0, len(data[i][j])):
                for l in range(0,len(data[i][j][k])):
                    if data[i][j][k][l] == np.array(2).astype(np.float16):
                        data[i][j][k][l] = np.array(1).astype(np.float16)   
    return data

def resize_to_1D(data):
    temp = []
    for i in range(0,len(data)):
        for j in range(0,len(data[i])):
            temp.append(data[i][j])
    return temp

def delete_null(automatic, corrected):
    #Find scans which have all class 0 in the automatic and corrected scans
    #Remove these from the automatic, corrected, and feature data arrays
    automatic_reduced = [] # np.array([], dtype = object)
    corrected_reduced = [] # np.array([], dtype = object)

    for i in range(0, len(automatic)):
        #temp_a = []
        #temp_c = []
        temp_auto = resize_to_1D(automatic[i])
        temp_corr = resize_to_1D(corrected[i])
        if (sum(temp_auto) + sum(temp_corr) != 0): 
            automatic_reduced.append(automatic[i])
            corrected_reduced.append(corrected[i])           
    return automatic_reduced, corrected_reduced

###---
    #Current working directory (cwd) should be the directory in which the first MRI folder is located
###---

#Ask whether to merge classes 1 and 2
ans_merge = input("Merge classes 1 and 2? (y/n) -> ")
merged = ""
if ans_merge == 'y':
    merged = "_merged"
    
ans_reduced = input("Remove data which is all background? (y/n) -> ")
reduced = ""
if ans_reduced == 'y':
    reduced = "_reduced"

for i in range(0,len(name)):
    for scan in scans:
        data_filepaths.append(cwd+filepath_old_baseline+'\\scan'+str(scan)+'\\'+name[i][0])
        data_filepaths.append(cwd+filepath_old_week12+'\\scan'+str(scan)+'\\'+name[i][0])    
        save_filepaths.append(cwd+filepath_baseline+'\\scan'+str(scan)+'\\'+name[i][1]+merged+reduced+'\\')
        save_filepaths.append(cwd+filepath_week12+'\\scan'+str(scan)+'\\'+name[i][1]+merged+reduced+'\\')    

# Check folders exist and create them if not
for path in save_filepaths:
    if not os.path.exists(path):
        os.makedirs(path)

# Convert into numpy array and create data array with all of them in
data = []   
for i in tqdm(range(0,len(data_filepaths))):
    sitkImage= sitk.ReadImage(data_filepaths[i])
    np_im = sitk.GetArrayFromImage(sitkImage)
    data.append(np_im)

if ans_reduced == 'y':
    auto = data[:16]
    corr = data[16:]
    data = []
    temp_auto = []
    temp_corr = []
    for i in tqdm(range(0,len(auto))):
        auto_red, corr_red = delete_null(auto[i], corr[i])

        temp_auto.append(auto_red)
        temp_corr.append(corr_red)
    data = temp_auto
    for i in range(0, len(temp_corr)):
        data.append(temp_corr[i]) 
        
if ans_merge == 'y':     
    data = merge_classes(data)

for i in tqdm(range(0,len(save_filepaths))): 
    print(len(save_filepaths))
    np.save(save_filepaths[i] + 'init.npy', data[i],allow_pickle = True)

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:44:55 2022

@author: jwluc
"""

import os
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk


filepath_old_baseline = "\MRI\MRI\BCAPPS\Baseline scans"
filepath_old_week12 = "\MRI\MRI\BCAPPS\Week 12 scans"
filepath_baseline = "\Data\Baseline scans"
filepath_week12 = "\Data\Week 12 scans"
name = [['FA14Saggital_TissueMap.mha','auto'],
        ['FA14Saggital_TissueMap_Corrected.mha','corrected'],
        ['FA14Saggital.mha','MRI']]
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

def delete_null(automatic, corrected, mri):
    #Find scans which have all class 0 in the automatic and corrected scans
    #Remove these from the automatic, corrected, and feature data arrays
    automatic_reduced = [] # np.array([], dtype = object)
    corrected_reduced = [] # np.array([], dtype = object)
    mri_reduced = []
    for i in range(0, len(automatic)):
        #temp_a = []
        #temp_c = []
        temp_auto = resize_to_1D(automatic[i])
        temp_corr = resize_to_1D(corrected[i])
        if (sum(temp_auto) + sum(temp_corr) != 0): 
            automatic_reduced.append(automatic[i])
            corrected_reduced.append(corrected[i])
            mri_reduced.append(mri[i])

    #automatic_reduced.append(temp_a)
    #corrected_reduced.append(temp_c)
           
    return automatic_reduced, corrected_reduced, mri_reduced

###---
    #Current working directory (cwd) should be the directory in which the first MRI folder is located
###---

r = ["","_reduced"]
m = [""]
for merged in m:
    for reduced in r:
        data_filepaths = []
        save_filepaths = []
        for i in range(0,len(name)):
            for scan in scans:
                data_filepaths.append(cwd+filepath_old_baseline+'\\scan'+str(scan)+'\\'+name[i][0])
                data_filepaths.append(cwd+filepath_old_week12+'\\scan'+str(scan)+'\\'+name[i][0])    
                save_filepaths.append(cwd+filepath_baseline+'\\scan'+str(scan)+'\\'+name[i][1]+merged+reduced+'\\')
                save_filepaths.append(cwd+filepath_week12+'\\scan'+str(scan)+'\\'+name[i][1]+merged+reduced+'\\')    
            #data_filepaths is a list off all .mha files to access
            #save_filepaths is a list of folders to save the numpy array files to
        #print(data_filepaths)
        #print(save_filepaths)
        
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
        
        if reduced == '_reduced':
            auto = data[:16]
            corr = data[16:32]
            mri = data[32:]
            data = []
            temp_auto = []
            temp_corr = []
            temp_mri = []
            for i in tqdm(range(0,len(auto))):
                auto_red, corr_red, mri_red = delete_null(auto[i], corr[i], mri[i])
        
                temp_auto.append(auto_red)
                temp_corr.append(corr_red)
                temp_mri.append(mri_red)
            data = temp_auto
            for i in range(0, len(temp_corr)):
                data.append(temp_corr[i])
            for i in range(0, len(temp_corr)):
                data.append(temp_mri[i])
                
            
                
        if merged == '_merged':     
            data = merge_classes(data)
        
        for i in tqdm(range(0,len(save_filepaths))): 
            np.save(save_filepaths[i] + 'init.npy', data[i],allow_pickle = True)

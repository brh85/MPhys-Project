# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 12:12:44 2022

@author: rory_
"""

import os
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

filepaths = []
filepath_baseline = "\Data\Baseline scans"
filepath_week12 = "\Data\Week 12 scans"
name = [['FA14Saggital_TissueMap.mha','auto'],
        ['FA14Saggital_TissueMap_Corrected.mha','corrected']]
scans=[1,3,6,7,10,12,13,19]
data_possibilities = [0,1]
cwd = os.getcwd()

def plot_scan_slice(scan, title): # Requires input of the 113x512x512 array
    plt.imshow(scan[10], interpolation='none')
    plt.title(title)
    plt.show()

#Ask whether to use data that merged classes 1 and 2
ans_merge = input("Merge classes 1 and 2? (y/n) -> ")
merged = ""
if ans_merge == 'y':
    merged = "_merged"
    
#Ask whether to use data with background slices removed
ans_reduced = input("Remove data which is all background? (y/n) -> ")
reduced = ""
if ans_reduced == 'y':
    reduced = "_reduced"

# Access the data filepath - note the output should be saved in the same folder so no need for separate save_filepaths array
for i in range(0,len(name)):
    for scan in scans:    
        filepaths.append(cwd+filepath_baseline+'\\scan'+str(scan)+'\\'+name[i][1]+merged+reduced+'\\')
        filepaths.append(cwd+filepath_week12+'\\scan'+str(scan)+'\\'+name[i][1]+merged+reduced+'\\')    

# Load the original data into init_data array 32x113x512x512
init_data = []
for i in tqdm(range(0,len(filepaths))):
    init_data.append(np.load(filepaths[i] + 'init.npy'))
    #plot_scan_slice(init_data[0], "Initial")
    
# First order augmentations - just one type (rot or flip)
augmentations = ['rot', 'flip']
augmented_data = []
for i in range(0,len(filepaths)):
    temp_aug = []
    for j in range(0,len(augmentations)):
        if augmentations[j] == 'rot':
            for k in range(0,3):
                temp = []
                for l in range(0, len(init_data[i])):
                    temp.append(np.rot90(init_data[i][l], k+1))
                temp_aug.append(temp) 
                
        if augmentations[j] == 'flip':
            for k in range(0, 2):
                temp = []
                for l in range(0,len(init_data[i])):
                    temp.append(np.flip(init_data[i][l], k))
                temp_aug.append(temp)
        ###----- Room for more augmentations to be added here -----###
        
        
        ###--------------------------------------------------------###
    augmented_data.append(temp_aug)

# Save the augmented data
for i in tqdm(range(0,len(filepaths))):
    for j in range(0, len(augmented_data[i])):
        # First 3 are rotations (90,180,270), next 2 are flips (v,h)
        filename = ["rot90", "rot180", "rot270", "flipv", "fliph"]
        np.save(filepaths[i] + filename[j] + '.npy', augmented_data[i][j], allow_pickle = True)


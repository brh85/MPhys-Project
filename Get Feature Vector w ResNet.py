# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 14:46:33 2021

@author: jwluc
"""

import os
from tqdm import tqdm
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

###-----------------------------------SETUP------------------------------------###
cwd = os.getcwd()
scans_to_look_at = [1,3,6,7,10,12,13,19]
filepath_baseline = "\\Data\\Baseline scans"
filepath_week12 = "\\Data\\Week 12 scans"
data_filepaths = []
save_filepaths = []


ans_merge = input("Use data with class 1 and 2 merged? (y/n) -> ")
merged = ""
if ans_merge == 'y':
    merged = "_merged"
    
ans_reduced = input("Use data with backgrounds removed? (y/n) -> ")
reduced = ""
if ans_reduced == 'y':
    reduced = "_reduced"


#select where to save data
for scan in scans_to_look_at:
    data_filepaths.append(cwd+filepath_baseline+'\\scan'+str(scan)+'\\auto'+merged+reduced+'\\')
    data_filepaths.append(cwd+filepath_week12+'\\scan'+str(scan)+'\\auto'+merged+reduced+'\\')    
    save_filepaths.append(cwd+filepath_baseline+'\\scan'+str(scan)+'\\auto_feat'+merged+reduced+'\\')
    save_filepaths.append(cwd+filepath_week12+'\\scan'+str(scan)+'\\auto_feat'+merged+reduced+'\\')    
  
for path in save_filepaths:
    if not os.path.exists(path):
        os.makedirs(path)  

channels = 3 #(rgb number)
rgb_mean = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]
A = []

###----------------LOADING MODEL--------------------###
model = torch.hub.load('pytorch/vision:v0.10.0','resnet18',pretrained = True)
model.float().eval()


for i in tqdm(range(0,len(data_filepaths))): # 16 iterations
    ###-----GETTING THE TENSOR OBJECT FROM THE NUMPY ARRAY------###
    temp = np.load(data_filepaths[i]+'init.npy', allow_pickle = True).astype(np.float32)
    feature_vectors = []
    for j in range(0,len(temp)): # 113 or z iterations
        img = []
        for k in range(0,channels): # 3 iterations
            img.append(temp[j])
        # img should now be a 3x512x512 array
    
        #-NORMALISING DATA TO BE BETWEEN 0 AND 1-#
        max_val = np.max(img)
        if max_val > 0:
            img = img/max_val
        
        #---------------CONVERT THE NUMPY ARRAY TO PYTORCH TENSOR-------------------------#
        normalise = transforms.Normalize(mean=rgb_mean, std=rgb_std)
        tensor = torch.from_numpy(np.array(img)).unsqueeze(0)
        del(img)
        ###---------------------------------------------------------###

        ###----------------------------------------------------------------------###
        if torch.cuda.is_available():
            tensor = tensor.to('cuda')
            model.to('cuda')
            print('Using GPU.')
        ###-------------------------------------------------###
        
        ###----------OBTAIN FEATURE VECTOR---------------###
    
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        output = feature_extractor(tensor)
        out = output.detach().numpy()
        feature_vectors.append(out[0])       
    #print("Feature vector: " + str(len(feature_vectors)), str(len(feature_vectors[0])), str(len(feature_vectors[0][0])), str(len(feature_vectors[0][0][0])))

    ###---------SAVE FEATURE VECTORS-------------------------------###

    temp_2 = []
    for j in range(0,len(feature_vectors)):
        temp = []
        for k in range(0,len(feature_vectors[j])):
            temp.append(feature_vectors[j][k][0][0])
        temp_2.append(temp)
    print("To save: " + str(len(temp_2)), str(len(temp_2[0])))
    np.save(save_filepaths[i] + 'init.npy', temp_2)
    # This should save feature vectors for a scan as a 113x512 array (or zx512 if reduced)

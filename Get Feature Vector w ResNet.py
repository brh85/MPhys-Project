# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 14:46:33 2021

@author: jwluc
"""

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 
import scipy
import torch
import cv2
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from torchvision import transforms

###-----------------------------------SETUP------------------------------------###
cwd = os.getcwd()
scans_to_look_at = [3,6,7,10,12,13,19]    #numbers of patient data to use

# select data type to use: 
#filename = '\FA14Saggital_numpy'
#filename = '\FA14Saggital_TissueMap_Corrected_numpy'
filename = '\FA14Saggital_TissueMap_numpy'

# select data either 'Baseline' or 'Week12'
b_or_w = ['Baseline','Week 12']

#select where to save data
date='02-11-21'
save_folder = '\Results\\'+date+'\\Feature Vectors'+filename+'\\'
if not os.path.exists(cwd+save_folder):
    os.makedirs(cwd+save_folder)

z=113
channels = 3 #(rgb number)
rgb_mean = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]



for B_or_W in b_or_w:
    filepath_base = [cwd+'\Data\\'+B_or_W+' Scans\scan'+str(scans_to_look_at[i])+filename for i in range(0,len(scans_to_look_at))]
    for scan in scans_to_look_at:
        filepath = cwd+'\Data\\'+B_or_W+' Scans\scan'+str(scan)+filename
    ###----------------------------------------------------------------------------####
        
        
        ###-----GETTING THE TENSOR OBJECT FROM THE NUMPY ARRAY------###
        A = []
        for i in range(0,z):
            temp = []
            for j in range(0,channels):
                temp.append(np.load(filepath+filename[:-5]+'plane'+str(i)+'.npy').astype(np.float32))
            A.append(temp)
        ##########################################
        #converting tissue at values 1 and 2 to both be 1    
        if filename != 'FA14Saggital_numpy':
            for i in tqdm(range(0,len(A))):
                for j in range(0,len(A[0])):
                    for k in range(0,len(A[0][0])):
                        for l in range(0,len(A[0][0][0])):
                            if A[i][j][k][l] == np.array(2).astype(np.float16):
                                A[i][j][k][l] = np.array(1).astype(np.float16)
        
        ##########################################
        #-NORMALISING DATA TO BE BETWEEN 0 AND 1-#
        
        max_val = np.max(A)
        A = A/max_val
        
        #---------------CONVERT THE NUMPY ARRAY TO PYTORCH TENSOR-------------------------#
        tensor = torch.from_numpy(np.array(A))
        del(A)
        ###---------------------------------------------------------###
        
        
        ###-------NORMALISING DATA TO THE MEAN AND STD OF THE RESNET MODEL-------###
        t = transforms.Normalize(mean=rgb_mean, std=rgb_std)
        tensor = t(tensor)
        print(tensor.shape)
        ###----------------------------------------------------------------------###
        
        
        ###----------------LOADING MODEL--------------------###
        model = torch.hub.load('pytorch/vision:v0.10.0','resnet18',pretrained = True)
        model.float().eval()
        # move the input and model to GPU for speed if available
        '''
        if torch.cuda.is_available():
            tensor = tensor.to('cuda')
            model.to('cuda')
        '''
        ###-------------------------------------------------###
        
        
        ###----------OBTAINING FEATURE VECTOR---------------###

        
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        output = feature_extractor(tensor)
        print(output.shape)
        

        ###-------------------------------------------------###
        '''
        output = model(tensor)
        
        print(output[0])
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        print(probabilities)
        '''
        ###---------SAVE FEATURE VECTORS-------------------------------###
        save_filepath = cwd + save_folder + filename[1:] + '_'+ B_or_W + '_scan' + str(scan)
        print(save_filepath)
        out = output.detach().numpy()
        temp = [[] for i in range(0,len(out))]
        for i in range(0,len(out)):
            for j in range(0,len(out[i])):
                temp[i].append(out[i][j][0][0])
        np.save(save_filepath,temp)
    


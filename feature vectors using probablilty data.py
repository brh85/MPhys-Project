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


r = ["","_reduced"]
m = [""]

data_filepaths = []
save_filepaths = []
mri_filepaths = []
wmri = "_wMRI"


for merged in m:
    for reduced in r:
        data_filepaths = []
        save_filepaths = []
        # Select where to save data
        for scan in scans_to_look_at:
            data_filepaths.append(cwd+filepath_baseline+'\\scan'+str(scan)+'\\prob'+merged+reduced+'\\')
            data_filepaths.append(cwd+filepath_week12+'\\scan'+str(scan)+'\\prob'+merged+reduced+'\\')    
            save_filepaths.append(cwd+filepath_baseline+'\\scan'+str(scan)+'\\prob_feat'+merged+reduced+wmri+'\\')
            save_filepaths.append(cwd+filepath_week12+'\\scan'+str(scan)+'\\prob_feat'+merged+reduced+wmri+'\\')    
            mri_filepaths.append(cwd+filepath_baseline+'\\scan'+str(scan)+'\\MRI'+merged+reduced+'\\')
            mri_filepaths.append(cwd+filepath_week12+'\\scan'+str(scan)+'\\MRI'+merged+reduced+'\\')                
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
            temp = os.listdir(data_filepaths[i])
            temp_mri = os.listdir(mri_filepaths[i])
            temp_mri_data = []
            temp_data = []
            augmented_filenames = []
            for j in temp:
                temp_data.append(np.load(data_filepaths[i]+'\\'+j).astype(np.float32))
                #now temp data has size j x z x 3 x 512 x 512 for prob data
                augmented_filenames.append(j[:-4])
            for j in temp_mri:
                temp_mri_data.append(np.load(mri_filepaths[i]+'\\'+j).astype(np.float32))
            print(len(temp_mri_data),len(temp_mri_data[0]),len(temp_data[0][0]),len(temp_data[0][0]))
            
            feature_vectors = []
            for n in tqdm(range(0, len(temp_data))): 
                temp_feat = []
                for j in range(0,len(temp_data[n])): # 113 or z iterations
                    img = [temp_data[n][j][1],temp_data[n][j][2],temp_mri_data[n][j]]
                    # img should now be a 3x512x512 array
                    ###--------NORMALISING DATA TO BE BETWEEN 0 AND 1-----------###
                    for k in range(0,len(img)):
                        max_val = np.max(img[k])
                        if max_val > 0:
                            img[k] = img[k]/max_val
                    #--------CONVERT THE NUMPY ARRAY TO PYTORCH TENSOR----------###
                    normalise = transforms.Normalize(mean=rgb_mean, std=rgb_std)
                    tensor = torch.from_numpy(np.array(img)).unsqueeze(0)
                    del(img)
                    ###---------------------------------------------------------###
                    '''
                    if torch.cuda.is_available():
                        tensor = tensor.to('cuda')
                        model.to('cuda')
                        print('Using GPU.')
                    '''      
                    ###----------------OBTAIN FEATURE VECTOR--------------------###
                
                    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
                    output = feature_extractor(tensor)
                    out = output.detach().numpy()
                    temp_feat.append(out[0])
                    # temp_feat should be zx512x1x1
                feature_vectors.append(temp_feat)
            print("Feature vector: " + str(len(feature_vectors)), str(len(feature_vectors[0])), str(len(feature_vectors[0][0])), str(len(feature_vectors[0][0][0])))
        
            ###---------SAVE FEATURE VECTORS-------------------------------###
            temp = []
            for j in range(0,len(feature_vectors)): #6
                temp_2 = []
                for k in range(0,len(feature_vectors[j])): #z
                    temp_3 = []
                    for l in range(0,len(feature_vectors[j][k])): #512
                        temp_3.append(feature_vectors[j][k][l][0][0])
                    temp_2.append(temp_3)
                temp.append(temp_2)
            print("To save: " + str(len(temp)), str(len(temp[0])),str(len(temp[0][0])))
        
            for j in range(0,len(augmented_filenames)):
                np.save(save_filepaths[i] + augmented_filenames[j] + '.npy', temp_2)
            # This should save feature vectors for a scan as a 113x512 array (or zx512 if reduced)
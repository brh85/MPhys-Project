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
import SimpleITK as sitk
from PIL import Image, ImageEnhance



###-----------------------------------SETUP------------------------------------###
cwd = os.getcwd()
scans_to_look_at = [1,3,6,7,10,12,13,19]
filepath_baseline = "\\Data\\Baseline scans"
filepath_week12 = "\\Data\\Week 12 scans"
filepath_old_baseline = "\\Data_old\\Baseline scans"
filepath_old_week12 = "\\Data_old\\Week 12 scans"

data_filepaths = []
save_filepaths = []
mri_filepaths = []

ans_mri = input("Use MRI image to calculate the feature vector? -> ")
wmri = ""
if ans_mri == 'y':
    wmri = "_wMRI"
    
ans_augmented = input("Use augmented (flipped and rotated) data? -> ")
augmented = ""
if ans_augmented == 'y':
    augmented = "_augmented"

ans_merge = input("Use data with class 1 and 2 merged? (y/n) -> ")
merged = ""
if ans_merge == 'y':
    merged = "_merged"
    
ans_reduced = input("Use data with all-background slices removed? (y/n) -> ")
reduced = ""
if ans_reduced == 'y':
    reduced = "_reduced"


# Select where to save data
for scan in scans_to_look_at:
    data_filepaths.append(cwd+filepath_baseline+'\\scan'+str(scan)+'\\auto'+merged+reduced+'\\')
    data_filepaths.append(cwd+filepath_week12+'\\scan'+str(scan)+'\\auto'+merged+reduced+'\\')    
    mri_filepaths.append(cwd+filepath_baseline+'\\scan'+str(scan)+'\\MRI'+reduced)
    mri_filepaths.append(cwd+filepath_week12+'\\scan'+str(scan)+'\\MRI'+reduced)
    save_filepaths.append(cwd+filepath_baseline+'\\scan'+str(scan)+'\\auto_feat'+merged+reduced+wmri+'\\')
    save_filepaths.append(cwd+filepath_week12+'\\scan'+str(scan)+'\\auto_feat'+merged+reduced+wmri+'\\')    
  
for path in save_filepaths:
    if not os.path.exists(path):
        os.makedirs(path)  


# Normalise an array (any dimension) to be between 0 and 1
def normalise_to1(array):
    maximum = np.amax(array)
    for i in range(0,len(array)):
        array[i] /= maximum 
    return array


channels = 3 #(rgb number)
rgb_mean = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]
A = []

###----------------LOADING MODEL--------------------###
model = torch.hub.load('pytorch/vision:v0.10.0','resnet18',pretrained = True)
model.float().eval()


for i in tqdm(range(0,len(data_filepaths))): # 16 iterations
    ###-----GETTING THE TENSOR OBJECT FROM THE NUMPY ARRAY------###
    if (ans_augmented == 'y'):
        temp = os.listdir(data_filepaths[i])
        temp_data = []
        augmented_filenames = []
        for j in temp: # should be 6 iterations
            temp_data.append(np.load(data_filepaths[i]+'\\'+j).astype(np.float32))
            augmented_filenames.append(j[:-4])
            
        # Space for loading augmented (flipped and rotated) mri scans
         
        
    else:
        temp_data = []
        mri_scan = []
        # For now replace init.npy with br12con18.npy and create new feature vectors to run the program as usual
        augmented_filenames = ['br12con18']
        temp_data.append(np.load(data_filepaths[i]+'\\init.npy').astype(np.float32))
        mri_scan.append(np.load(mri_filepaths[i]+'\\init.npy').astype(np.float32))
        # mri_scan should be a 1xzx512x512 array

        ans_augment_mri = input("Augment the MRI image in the R channel? -> ")
        if ans_augment_mri == 'y':
            mri_scan_aug = []
            for j in range(0,len(mri_scan)):
                temp = []
                for k in range(0,len(mri_scan[j])):
                    img = Image.fromarray(np.uint8(normalise_to1(mri_scan[j][k])*255))
                    contraster = ImageEnhance.Contrast(img)
    
                    mri_contrast = contraster.enhance(1.8)
    
                    brightener = ImageEnhance.Brightness(mri_contrast)
    
                    mri_bright_contrast = brightener.enhance(1.2)
                    temp.append(mri_bright_contrast)
                mri_scan_aug.append(temp)
            mri_scan = mri_scan_aug
            del(mri_scan_aug)


    # temp_data is now a 6xzx512x512 array
    
    #temp = np.load(data_filepaths[i]+'init.npy', allow_pickle = True).astype(np.float32)
    feature_vectors = []
    for n in tqdm(range(0, len(temp_data))): # 6 (or 1) iterations
        temp_feat = []
        for j in range(0,len(temp_data[n])): # 113 or z iterations
            img = []
            if (ans_mri == 'y'):
                # Create 3x512x512 array with MRI, class 1, class 2 as the 3 parts
                #Create binary map of class 1 and 2
                binary_class_maps = []
                for a in range(1,3):
                    binary_class_temp_2 = []
                    for b in range(0,len(temp_data[n][j])):
                        binary_class_temp = []
                        for c in range(0,len(temp_data[n][j][b])):
                            if (temp_data[n][j][b][c] == a):
                                binary_class_temp.append(1)
                            else:
                                binary_class_temp.append(0)
                        binary_class_temp_2.append(binary_class_temp)
                    binary_class_maps.append(binary_class_temp_2)
                
                mri_max_val = np.max(mri_scan[n][j])
                if mri_max_val > 0:
                    mri_scan[n][j] /= mri_max_val
                img.append(mri_scan[n][j])
                img.append(binary_class_maps[0])
                img.append(binary_class_maps[1])
            else:
                for k in range(0,channels): # 3 iterations
                    img.append(temp_data[n][j])

                max_val = np.max(img)
                if max_val > 0:
                    img = img/max_val
                
            # img should now be a 3x512x512 array
            
            #--------CONVERT THE NUMPY ARRAY TO PYTORCH TENSOR----------###
            normalise = transforms.Normalize(mean=rgb_mean, std=rgb_std)
            tensor = torch.from_numpy(np.array(img)).unsqueeze(0)
            del(img)
            ###---------------------------------------------------------###
    
            if torch.cuda.is_available():
                tensor = tensor.to('cuda')
                model.to('cuda')
                print('Using GPU.')
            
            ###----------------OBTAIN FEATURE VECTOR--------------------###
        
            feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
            output = feature_extractor(tensor.float())
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
    #print("To save: " + str(len(temp)), str(len(temp[0])),str(len(temp[0][0])))

    for j in range(0,len(augmented_filenames)):
        np.save(save_filepaths[i] + augmented_filenames[j] + '.npy', temp_2)
    # This should save feature vectors for a scan as a 113x512 array (or zx512 if reduced)

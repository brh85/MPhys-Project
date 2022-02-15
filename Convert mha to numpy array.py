# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 12:47:40 2021

@author: Rory Harris
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 11:52:38 2021

@author: jwluc
"""

import os
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk

data_filepaths = []
save_filepaths = []
cwd = os.getcwd()
print(cwd)

###---
    #Current working directory (cwd) should be the directory in which the first MRI folder is located
###---

filepath_baseline = "\Data\Baseline scans"
filepath_week12 = "\Data\Week 12 scans"

name = 'FA14Saggital_TissueMap.mha'
#name = 'FA14Saggital_TissueMap_Corrected.mha'
scans=[1,3,6,7,10,12,13,19]

for scan in scans:
    data_filepaths.append(cwd+filepath_baseline+'\\scan'+str(scan)+'\\'+name)
    data_filepaths.append(cwd+filepath_week12+'\\scan'+str(scan)+'\\'+name)    
    save_filepaths.append(cwd+filepath_baseline+'\\scan'+str(scan)+'\\'+name[:-4]+'_numpy\\'+name[:-4])
    save_filepaths.append(cwd+filepath_week12+'\\scan'+str(scan)+'\\'+name[:-4]+'_numpy\\'+name[:-4])    
#data_filepaths is a list off all .mha files to access
#save_filepaths is a list of folders to save the numpy array files to

#check folders exist
for path in save_filepaths:
    if not os.path.exists(path[:-(len(name[:-4])+1)]):
        os.makedirs(path[:-(len(name[:-4])+1)])

   
for a in tqdm(range(0,1)):#len(data_filepaths))):
    sitkImage= sitk.ReadImage(data_filepaths[a])
    np_im = sitk.GetArrayFromImage(sitkImage)
    print(len(np_im),len(np_im[0]),len(np_im[0][0]))
    for k in range(0,5):
        arr = np_im[k]
        #Use below to print the numpy array of one of the 113 slices
        print(sum(arr))
        '''
        #for i in range(0,len(np_im[0])):
            #for j in range(0,len(np_im[0][0])):
                #arr.append([i,j,k,np_im[k][j][i]])
        for i in range(len(arr)):
            for j in range(len(arr[0])):
                if k==0:
                    print(arr[i][j])
        #np.save(save_filepaths[a]+'_plane'+str(k),arr)
        '''

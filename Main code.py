# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 20:29:12 2022

@author: rory_
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 12:43:10 2021

@author: rory_
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:54:00 2021

@author: jwluc
"""

import os
from tqdm import tqdm
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm 
#import scipy
#import torch
#import cv2
#import pydicom
#import SimpleITK as sitk
#import itertools
import random
from datetime import datetime

def get_metrics_and_errors(results, test_set, test_set_dice_score, test_set_dice_score_numerical_value, beta, data_possibilities, metric, classifier, k, folder_temp):
    #returns correct metric and error of test data given metric name 
    score = get_metric(results, test_set_dice_score, test_set_dice_score_numerical_value, beta, data_possibilities, 1, metric, k, folder_temp)
    error_range_acc = bootstrapping(test_set, test_set_dice_score, test_set_dice_score_numerical_value, classifier, iterations_of_bootstrapping, beta, data_possibilities, metric, k, folder_temp)
    err_acc = np.sqrt((error_range_acc[1]-error_range_acc[0])**2)
    return [metric,score,err_acc]

def get_metric(results, test_set_dice_score, test_set_dice_score_numerical_value, beta, data_possibilities, plot, metric, k, folder_temp):
    # plot = 1 (plot) or = 0 (no plot)
    if metric == 'F-beta':
        score = F_beta(results, test_set_dice_score, beta, data_possibilities)
    if metric == 'Precision':
        score = Precision(results, test_set_dice_score, data_possibilities)
    if metric == 'Recall':
        score = Recall(results, test_set_dice_score, data_possibilities)
    if metric == 'Accuracy':
        score = Acc(results, test_set_dice_score)
    if metric == 'AUC':
        score = plot_ROC_curve(results, test_set_dice_score_numerical_value, data_possibilities, plot, k, folder_temp)
    return score

def weighted_mean(x,err):
    a = 0
    b = 0
    for i in range(0,len(x)):
        a += x[i]/(err[i]**2)
        b += 1/(err[i]**2)
    return a/b

def standard_error_on_mean(err):
    a = 0
    for i in range(0,len(err)):
        a += 1/(err[i]**2)
    return np.sqrt(1/a)        

def plot_ROC_curve(predicted_data, dice_score_values, data_possibilities, plot, k, folder_temp):    
    X,Y,th = sklearn.metrics.roc_curve(predicted_data, dice_score_values)
    if plot == 1:
        #plt.scatter(X,Y,marker='.',c='blue')    
        plt.plot(X,Y)
        plt.xlim(min(X),max(X))
        plt.ylim(min(Y),max(Y))
        plt.plot([0,1],[0,1],ls="--",c='green')
        plt.ylabel('True positive rate')
        plt.xlabel('False positive rate')
        plt.savefig(folder_temp + '\\k='+str(k)+'_ROC_curve')
        plt.show() 
    area = sklearn.metrics.roc_auc_score(predicted_data, dice_score_values)
    return area
            
def get_confusion_matrix_data(predicted_data,actual_data,data_possibilities):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for i in range(0,len(predicted_data)):
        if predicted_data[i] == actual_data[i]:
            if predicted_data[i] == data_possibilities[1]: #good
                true_pos += 1
            if predicted_data[i] == data_possibilities[0]: #bad
                true_neg += 1
        else:
            if predicted_data[i] == data_possibilities[1]: #good
                false_pos += 1
            if predicted_data[i] == data_possibilities[0]: #bad
                false_neg += 1
    #print(true_pos,true_neg,false_pos,false_neg)
    return true_pos,true_neg,false_pos,false_neg

def bootstrapping(test_set,test_set_dice_scores,test_set_dice_scores_numerical_values,model,iterations,beta,data_possibilities,metric,k,folder_temp):
    #metric is for example accuracy or F-beta
    bootstrapping_data = []
    for i in range(0,iterations):
        new_data = []
        actual_result = []
        actual_result_values = []
        for j in range(0,len(test_set)):
            x = np.random.randint(0,len(test_set)-1)
            new_data.append(test_set[x])
            actual_result.append(test_set_dice_scores[x])
            actual_result_values.append(test_set_dice_scores_numerical_values[x])
        new_result = model.predict(new_data)
        score = get_metric(new_result, actual_result, actual_result_values, beta, data_possibilities, 0, metric, k, folder_temp)
        bootstrapping_data.append(score)
    bootstrapping_data.sort()
    confidence_interval = [bootstrapping_data[round(0.025*len(bootstrapping_data))],bootstrapping_data[round(len(bootstrapping_data)-(0.025*len(bootstrapping_data)))]]
    
    #plotting
    plt.hist(bootstrapping_data,bins=25)
    plt.ylabel('number of scores')
    plt.xlabel(metric)
    plt.savefig(folder_temp + '\\k=' + str(k) + ' - ' + metric + ' - iterations=' + str(iterations))
    plt.show()
    return confidence_interval

def Acc(predicted_data,actual_data):
    accuracy = 0
    for i in range(0,len(predicted_data)):
        if predicted_data[i] == actual_data[i]:
            accuracy += 1
    return accuracy*100/len(predicted_data)

def Precision(predicted_data,actual_data,data_possibilities):
    true_pos, true_neg, false_pos, false_neg = get_confusion_matrix_data(predicted_data,actual_data,data_possibilities)
    if true_pos == 0:
        return 0
    return true_pos/(true_pos+false_pos)

def Recall(predicted_data,actual_data,data_possibilities):
    true_pos, true_neg, false_pos, false_neg = get_confusion_matrix_data(predicted_data,actual_data,data_possibilities)
    if true_pos == 0:
        return 0    
    return true_pos/(true_pos+false_neg)

def F_beta(predicted_data,actual_data,beta,data_possibilities):
    #p: precision, r: recall. Predicted data and actual data must be the same size. At the moment will not work for multi-class
    true_pos, true_neg, false_pos, false_neg = get_confusion_matrix_data(predicted_data,actual_data,data_possibilities)
    p = Precision(predicted_data,actual_data,data_possibilities)
    r = Recall(predicted_data,actual_data,data_possibilities)
    if p == 0 or r == 0:
        return 0
    return (1 + (beta**2))*(p*r)/((beta*p)+r)

def Dice_Score(A,B):
    #Arguments must be location of numpy array files of the data, z is slice number 
    #A is the automatic segmentation B is corrected version
    #n_0 = 0
    #n_1 = 0
    temp_A = resize_for_svm(A)
    temp_B = resize_for_svm(B)
    TP, TN, FP, FN = get_confusion_matrix_data(temp_A, temp_B, data_possibilities)
    
    if TP>0:
        d_s = 2*TP/(2*TP+FP+FN)
    else:
        d_s = 0
    #if TN>0:
     #   d_s_1 = 2*TN/(2*TN+FP+FN)
    #else:
     #   d_s_1 = 0
    #if TP + TN > 0:
     #   d_s = (d_s_0 + d_s_1)/2
    #else:
    #    d_s = d_s_0 + d_s_1
    #print('Class 0: ' + str(d_s_0))# + ', n: ' + str(n_0))
    #print('Class 1: ' + str(d_s_1))# + ', n: ' + str(n_1))
    print(d_s)
    return d_s

def get_Dice_Score(val,training_data,im_data,cor_im_data,dice_score_save_loc,data_possibilities):
    Dice_Scores = []
    dice_score_values = []
    dice_plot = []
    for i in tqdm(range(0,len(training_data))):
        temp = []
        temp_2 = []
        for j in range(0,len(training_data[i])):
            print(len(training_data[i]))
            print(len(im_data[i][j]))
            print(len(cor_im_data[i][j]))

            d_s = Dice_Score(im_data[i][j], cor_im_data[i][j])
            dice_plot.append(d_s)
            temp_2.append(d_s)
            if d_s > val:
                temp.append(data_possibilities[1])
            if d_s <= val:
                temp.append(data_possibilities[0])
        Dice_Scores.append(temp)
        dice_score_values.append(temp_2)
    np.save(cwd + dice_score_save_loc + "\\Dice_Scores",Dice_Scores)
    np.save(cwd + dice_score_save_loc + "\\Dice_Score_Values",dice_score_values)
    np.save(cwd + dice_score_save_loc + "\\Dice_Plot",dice_plot)
    return Dice_Scores, dice_score_values, dice_plot

###---------GET LOCATION OF DATA--------------###

def get_Tissue_Map_feature_vector(patients, ans_reduced):
    #function which returns the Tissue Map feature vectors from the selected patients 
    reduced = ''
    Red = ''
    if ans_reduced == 'y':
        reduced = '_reduced'
        Red = ' Reduced'
    
    filepath_Tissue_Map_Base = [cwd + '\\Results\\'+date+'\\Feature Vectors' + Red + '\\FA14Saggital_TissueMap_numpy\\FA14Saggital_TissueMap_numpy_Baseline_scan' + str(scans_to_look_at[i]) + reduced +'.npy' for i in range(0,len(scans_to_look_at))]
    filepath_Tissue_Map_Week12 = [cwd + '\\Results\\'+date+'\\Feature Vectors' + Red + '\\FA14Saggital_TissueMap_numpy\\FA14Saggital_TissueMap_numpy_Week 12_scan' + str(scans_to_look_at[i]) + reduced +'.npy' for i in range(0,len(scans_to_look_at))]
    filepaths = []
    for i in range(0,len(filepath_Tissue_Map_Base)):
        filepaths.append(filepath_Tissue_Map_Base[i])
        filepaths.append(filepath_Tissue_Map_Week12[i])
    data = []
    for filepath in filepaths:
        temp = np.load(filepath)
        data.append(temp)
    return data

def get_Tissue_Map_image_data(patients, ans_reduced):
    Data = []
    reduced = ''
    if ans_reduced == 'y':
        reduced = '_reduced\\'
    
    for scan in patients:      
        temp = os.listdir(cwd+'\\Data\\Baseline Scans\\scan'+str(scan)+'\\FA14Saggital_TissueMap_numpy'+reduced)
        print(temp)
        temp_2 = os.listdir(cwd+'\\Data\\Week 12 Scans\\scan'+str(scan)+'\\FA14Saggital_TissueMap_numpy'+reduced)
        data = []
        data_2 = []
        for z in temp:
            data.append(np.load(cwd+'\\Data\\Baseline Scans\\scan'+str(scan)+'\\FA14Saggital_TissueMap_numpy'+reduced+'\\'+z))#'FA14Saggital_TissueMap_plane'+str(z)))
        for z in temp_2:
            data_2.append(np.load(cwd+'\\Data\\Week 12 Scans\\scan'+str(scan)+'\\FA14Saggital_TissueMap_numpy'+reduced+'\\'+z))#'FA14Saggital_TissueMap_plane'+str(z)))
        # now data and data_2 are 113x512x512 image data numpy arrays - data is of baseline ordered scan 1-> 19 and data_2 is of week 12 data 
        # unless loading reduced files - then it is zx512x512
        Data.append(data)
        Data.append(data_2)
        
        
    return Data

def get_Tissue_Map_Corrected_image_data(patients, ans_reduced):
    Data = []
    reduced = ''
    if ans_reduced == 'y':
        reduced = '_reduced'
    
    for scan in patients:      
        temp = os.listdir(cwd+'\\Data\\Baseline Scans\\scan'+str(scan)+'\\FA14Saggital_TissueMap_Corrected_numpy'+reduced)
        temp_2 = os.listdir(cwd+'\\Data\\Week 12 Scans\\scan'+str(scan)+'\\FA14Saggital_TissueMap_Corrected_numpy'+reduced)
        data = []
        data_2 = []
        for z in temp:
            data.append(np.load(cwd+'\\Data\\Baseline Scans\\scan'+str(scan)+'\\FA14Saggital_TissueMap_Corrected_numpy'+reduced+'\\'+z))#'FA14Saggital_TissueMap_plane'+str(z)))
        for z in temp_2:
            data_2.append(np.load(cwd+'\\Data\\Week 12 Scans\\scan'+str(scan)+'\\FA14Saggital_TissueMap_Corrected_numpy'+reduced+'\\'+z))#'FA14Saggital_TissueMap_plane'+str(z)))
        # now data and data_2 are 113x512x512 image data numpy arrays - data is of baseline ordered scan 1-> 19 and data_2 is of week 12 data 
        # unless loading reduced files - then it is zx512x512
        Data.append(data)
        Data.append(data_2)
    return Data

def randomiser(a,b):
    # a and b same size - to be shuffled the same 
    done = []
    temp_a = []
    temp_b = []
    for i in range(0,len(a)):
        r = random.randint(0,len(a)-1)
        while r in done:
            r = random.randint(0,len(a)-1)
        temp_a.append(a[r])
        temp_b.append(b[r])
        done.append(r)
    return temp_a,temp_b

def split_data(data,k,n):
    Train = []
    Validation = []
    Test = []
    done = []
    for i in range(k,k+n):
        while i > len(data):
            i = i - len(data)
        Validation.append(data[i])
        done.append(i)
    for i in range(k+n,k+n+n):
        while i > len(data):
            i = i - len(data)
        Test.append(data[i])
        done.append(i)
    for i in range(0,len(data)):
        if i not in done:
            Train.append(data[i])
    return Train,Validation,Test

def resize_for_svm(data):
    temp = []
    for i in range(0,len(data)):
        for j in range(0,len(data[i])):
            temp.append(data[i][j])
    return temp

def load_saved_dice_scores(location):
    list_of_files = os.listdir(cwd + location)
    temp = []
    for file in list_of_files:
        temp.append(np.load(cwd + location + "\\"+ file))
    return temp[1],temp[2],temp[0] #dice_scores, dice_score_values, dice_plot

def write_results_to_file(Results,av,folder,hyperparameter,mean,confusion_matrices,metric_for_validation):
    my_results_file = open(folder + '\\results.txt','w')
    my_results_file.write('Mean of dice score of data set: '+str(mean)+'\n')
    my_results_file.write('Metric optimised in validation stage: '+metric_for_validation+'\n')
    for i in range(0,len(confusion_matrices)):
        my_results_file.write(str(confusion_matrices[i])+'\n')
    for i in range(0,len(Results)):
        my_results_file.write('Model '+str(i)+', hyperparameter: '+str(hyperparameter[i])+'\n')
        for j in range(0,len(Results[0])):
            my_results_file.write(Results[i][j][0]+': '+str(round(Results[i][j][1],3))+u"\u00B1"+str(round(Results[i][j][2],3))+'\n')
        my_results_file.write('\n')
    my_results_file.write('Averages:\n')
    for i in range(0,len(av)):
        my_results_file.write(av[i][0]+': '+str(round(av[i][1],3))+u"\u00B1"+str(round(av[i][2],3))+'\n')
    my_results_file.close()

def save_reduced(automatic_reduced, corrected_reduced, feature_reduced):
    AC = ['', 'Corrected']
    BW = ['Baseline', 'Week 12']
    
    # Save segmentations
    for i in range(0,len(scans_to_look_at)): #scan in scans_to_look_at:
        for A_or_C in AC:
            for j in range(0,len(BW)): #B_or_W in BW:
                filepath = cwd+'\\Data\\'+BW[j]+' Scans\\scan'+str(scans_to_look_at[i])+'\\FA14Saggital_TissueMap_'+ A_or_C +'numpy_reduced\\'
                
                temp = []
                if A_or_C == '':
                    temp.append(automatic_reduced[2*i + j])
                
                    # FA14Saggital_TissueMap_plane0
                else:
                    temp.append(corrected_reduced[2*i + j])
               
                for z in range(0,len(temp)):
                    print(len(temp))
                    print(len(temp[0]))
                    print(len(temp[0][0]))
                    temp_2 = []
                    temp_2.append(temp[z])
                    # for i in range(0,len(corrected_reduced)):
                    #    temp.append(corrected_reduced[i])
                    save_filepath = filepath + 'FA14Saggital_TissueMap_plane' + str(z)
                    if not os.path.exists(filepath):
                        os.makedirs(filepath)
                    np.save(save_filepath,temp_2)    
    
    # Save reduced feature vectors
    filename = '\\FA14Saggital_TissueMap_numpy_reduced'
    save_folder = '\Results\\'+date+'\\Feature Vectors Reduced\\FA14Saggital_TissueMap_numpy_reduced\\'

    if not os.path.exists(cwd+save_folder):
        os.makedirs(cwd+save_folder)
    
    for scan in scans_to_look_at:
        for B_or_W in BW:
            save_filepath = cwd + save_folder + filename + '_' + B_or_W + '_scan' + str(scan)
            temp = []
            print(save_filepath)
            for i in range(0,len(feature_reduced)):
                temp.append(feature_reduced[i])
            np.save(save_filepath,temp)
#"""
def delete_null(automatic, corrected, feature):
    #Find scans which have all class 0 in the automatic and corrected scans
    #Remove these from the automatic, corrected, and feature data arrays
    automatic_reduced = []
    corrected_reduced = []
    feature_reduced = []

    for i in tqdm(range(0, len(automatic))):
        temp_a = []
        temp_c = []
        temp_f = []
        for j in range(0,len(automatic[i])):
            temp_auto = resize_for_svm(automatic[i][j])
            temp_corr = resize_for_svm(corrected[i][j])
            #print(len(temp_auto))
            #print(sum(temp_auto))
            TP, TN, FP, FN = get_confusion_matrix_data(temp_auto, temp_corr, data_possibilities)
            #print('TN:' + str(TN) + ', TP:' + str(TP) + ', FN:' + str(FN) + ', FP:' + str(FP))
            if (TP + FP + FN) != 0:
                temp_a.append(automatic[i][j])
                temp_c.append(corrected[i][j])
                temp_f.append(feature[i][j])
            else:
                BW = ''
                if (i % 2) == 0: 
                     BW = 'B'
                else:
                     BW = 'W'
                #print('Scan: '+str(scans_to_look_at[round(i/2)])+ BW +', Slice'+str(j+1))
        print(len(temp_a))
        print(len(temp_a[0]))
        print(len(temp_a[0][0]))


        automatic_reduced.append(temp_a)
        corrected_reduced.append(temp_c)
        feature_reduced.append(temp_f)
    print(len(automatic_reduced))
    print(len(automatic_reduced[0]))
    #print(len(automatic_reduced[0][0]))
    #print(len(automatic_reduced[0][0][0]))
    
    save_reduced(automatic_reduced, corrected_reduced, feature_reduced)
                
    return automatic_reduced, corrected_reduced, feature_reduced
#"""
######------------PLOTTING HISTOGRAM OF DICE SCORES------------###

def plot_dice_scores(scores_plot):
    plt.hist(scores_plot,bins=20)
    plt.ylabel('number of scores')
    plt.xlabel('dice score')
    plt.show()
    scores = np.array(scores_plot)
    return scores.mean(),np.median(scores)

###-------------------------------------------------------------------------###

def main():
    # GET AUTOMATIC AND CORRECTED SEGMENTATIONS AND THE FEATURE VECTORS
    ans_1 = input("Remove all background segmentations? (y/n) -> ")
    ans_2 = ""
    if ans_1 == "y":
        ans_2 = input("Load files with background already removed? (y/n) -> ")

    
    image_data = get_Tissue_Map_image_data(scans_to_look_at, ans_2)
    Corrected_image_data = get_Tissue_Map_Corrected_image_data(scans_to_look_at, ans_2)
    feature_data = get_Tissue_Map_feature_vector(scans_to_look_at, ans_2) #patient data has size (14,113,512)
    
    #REMOVE THE SCANS WHICH ARE ALL BACKGROUND
    if ans_2 == "n":
        auto_reduced, corr_reduced, feat_reduced = delete_null(image_data, Corrected_image_data, feature_data)
    
    #16 scans - 2 x 8 patients at baseline and week 12, ordered: patient 1 baseline, patient 1 week 12, patient 2 baseline, patinet 2 week 12 etc.
    #Then usual feature vector list, with 113 slices for each data set
    
    ans_3 = input("Re-calculate dice scores (y/n) -> ")
    if ans_3 == "y":
        dice_scores, dice_score_values, dice_plot = get_Dice_Score(val, feat_reduced, auto_reduced, corr_reduced, dice_score_save_folder_location,data_possibilities)
    else:
        dice_scores, dice_score_values, dice_plot = load_saved_dice_scores(dice_score_save_folder_location)
        
    mean,median = plot_dice_scores(dice_plot)
    print(mean,median)
    ###---------------------------------------------------------###
    
    for metric_for_validation in metrics_for_validation:    #loop over all chosen metrics to be maximised in validation
        print('maximising',metric_for_validation,'in validation stage')
        folder_temp = folder + '\\' + metric_for_validation
        Results = []
        hyperparameters = []
        confusion_matrices = [] 
        averages = [[metric,[],[]] for metric in metrics]
        calculated_averages = [[metric,[],[]] for metric in metrics]
        Scores = []
        C_Vals = []
        #K-FOLD VALIDATION LOOP 
        #NOW 4-FOLD
        
        for k in tqdm(range(0,k_max)):
            
            best_score = 0
            c_best = 0
            Train,Validation,Test = split_data(feat_reduced,k,n)
            Train_image_data,Validation_image_data,Test_image_data = split_data(auto_reduced,k,n)
            Train_image_data_corr,Validation_image_data_corr,Test_image_data_corr = split_data(corr_reduced,k,n)
            Train_dice_Scores,Validation_dice_Scores,Test_dice_Scores = split_data(dice_scores,k,n)
            Train_dice_score_numerical_value,Validation_dice_score_numerical_value,Test_dice_score_numerical_value = split_data(dice_score_values,k,n)
        
            ###----------RESIZE THE ARRAYS TO BE USED IN THE SVM (2D to 1D array)------------------###
            
            Train_new = resize_for_svm(Train)
            Validation_new = resize_for_svm(Validation)
            Test_new = resize_for_svm(Test)
            
            Dice_new_Train = resize_for_svm(Train_dice_Scores)
            Dice_new_Validation = resize_for_svm(Validation_dice_Scores)
            Dice_new_Test = resize_for_svm(Test_dice_Scores)

            Train_dice_score_numerical_value = resize_for_svm(Train_dice_score_numerical_value)
            Validation_dice_score_numerical_value = resize_for_svm(Validation_dice_score_numerical_value)
            Test_dice_score_numerical_value = resize_for_svm(Test_dice_score_numerical_value)
            
            Train_new,Dice_new_Train = randomiser(Train_new,Dice_new_Train) #randomise training set 
            
            ###------------------------VALIDATION-------------------------------------------###
            scores = []
            c_vals = []
            for i in range(0,num_of_validation_iterations):
                for j in range(1,10):
                    c=j*(10**(i-(num_of_validation_iterations/2)))
                    classifier = svm.LinearSVC(C=c,max_iter=svm_iterations, dual = False)
                    classifier.fit(Train_new,Dice_new_Train)   
                    validate = classifier.predict(Validation_new)
                    score = get_metric(validate, Dice_new_Validation, Validation_dice_score_numerical_value, beta, data_possibilities, 0, metric_for_validation, k, folder_temp)
                    c_vals.append(c)
                    scores.append(score)
                    if score > best_score:
                        best_score = score
                        c_best = c
            
            ###--------------------TEST BEST MODEL--------------------------###
            
            classifier = svm.LinearSVC(C=c_best,max_iter=svm_iterations, dual = False)
            classifier.fit(Train_new,Dice_new_Train)   
            results = classifier.predict(Test_new)
            print(results)
            print(type(results))
            temp = []
            for metric in metrics:
                temp.append(get_metrics_and_errors(results, Test_new, Dice_new_Test, Test_dice_score_numerical_value, beta, data_possibilities, metric, classifier, k, folder_temp))
            Results.append(temp)
            hyperparameters.append(c_best)
            confusion_matrices.append(get_confusion_matrix_data(results, Dice_new_Test, data_possibilities))
            Scores.append(scores)
            C_Vals.append(c_vals)
        
        for i in range(0,len(Scores)):
            plt.plot(C_Vals[i],Scores[i],label='k='+str(i))
        plt.xlabel('C')
        plt.ylabel(metric_for_validation)
        plt.xscale('log')
        plt.yscale('linear')
        plt.show()
            
        for i in range(0,len(Results)):
            for j in range(0,len(Results[i])):
                for k in range(0,len(averages)):
                    if Results[i][j][0] == averages[k][0]:
                        averages[k][1].append(Results[i][j][1])
                        averages[k][2].append(Results[i][j][2])
        for i in range(0,len(averages)):
            calculated_averages[i][1] = weighted_mean(averages[i][1], averages[i][2])
            calculated_averages[i][2] = standard_error_on_mean(averages[i][2])
        write_results_to_file(Results, calculated_averages, folder_temp, hyperparameters, mean, confusion_matrices, metric_for_validation)
        
    # Rename file to remove unfinished name
    old = cwd + '\\Results\\Graphs\\' + date_and_time_now + 'unfinished'
    new = cwd + '\\Results\\Graphs\\' + date_and_time_now
    os.rename(old, new)

            
###-------------------------DEFINE VARIABLES--------------------------------###
now = datetime.now()
date_and_time_now = now.strftime("%d/%m/%Y %H:%M:%S")
date_and_time_now = date_and_time_now.replace(':', ';')
date_and_time_now = date_and_time_now.replace('/','-')
scans_to_look_at = [1,3,6,7,10,12,13,19]    #numbers of patient data to use
cwd = os.getcwd()
date = '02-11-21'
val = 0.958 #median of DSC with all background slices removed
num_of_validation_iterations = 6
n = 4
k_max = 4
beta = 1
iterations_of_bootstrapping = 10000 #(10000)
data_possibilities = [0,1] #['bad','good']
svm_iterations = 1000 #no. iterations when training svm
dice_score_save_folder_location = "\\Results\\New_Dice_Scores_val_" + str(val) #"\\Results\\Dice_Scores_val_mean"
metrics = ['Accuracy','Precision','Recall','F-beta','AUC']
metrics_for_validation = ['Accuracy','Precision','Recall','F-beta'] #metric which the validation stage maximises
folder = cwd + '\\Results\\Graphs\\' + date_and_time_now + 'unfinished'
for metric in metrics_for_validation:
    if not os.path.exists(folder + '\\' + metric):
        os.makedirs(folder + '\\' + metric)
if not os.path.exists(cwd + dice_score_save_folder_location):
    os.makedirs(cwd + dice_score_save_folder_location)
main()
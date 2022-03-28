# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:44:25 2022

@author: rory_
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 15:23:34 2022

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

def get_metrics_and_errors(results, test_set, test_set_dice_score, test_set_dice_score_numerical_value, beta, data_possibilities, metric, classifier, k, folder_temp):
    #returns correct metric and error of test data given metric name 
    score = get_metric(results, test_set_dice_score, test_set_dice_score_numerical_value, beta, data_possibilities, 1, metric, k, folder_temp)
    if SVM == 1:
        error_range_acc = bootstrapping(test_set, test_set_dice_score, test_set_dice_score_numerical_value, classifier, iterations_of_bootstrapping, beta, data_possibilities, metric, k, folder_temp)
        err_acc = np.sqrt((error_range_acc[1]-error_range_acc[0])**2)
        return [metric,score,err_acc]
    else:
        return [metric,score,0]
        

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
    if metric == 'mean_squared_error':
        score = sklearn.metrics.mean_squared_error(test_set_dice_score_numerical_value, results)
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
    #print(sklearn.metrics.confusion_matrix(actual_data,predicted_data))
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
    plt.ylabel('Number of scores')
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

def dice_from_jaccard_index(A,B):
    if merged == 0:
        JI = sklearn.metrics.jaccard_score(A,B, average = 'macro')
    else:
        JI = sklearn.metrics.jaccard_score(A,B,average = 'binary')
    return 2*JI/(1+JI)

def get_Dice_Score(val,training_data,im_data,cor_im_data,dice_score_save_loc,data_possibilities,ans_augmented):
    Dice_Scores = []
    dice_score_values = []
    dice_plot = []
    augmented = ''
    if ans_augmented == 'y':
        augmented = "_augmented"
    for i in tqdm(range(0,len(training_data))): #16 iterations
        temp_a = []
        temp_b = []
        for j in range(0,len(training_data[i])): #6 iterations
            temp = []
            temp_2 = []
            for k in range(0,len(training_data[i][j])): #113 iterations
                #d_s = Dice_Score(im_data[i][j], cor_im_data[i][j], data_possibilities)
                d_s = dice_from_jaccard_index(resize_2d_array(im_data[i][j][k]), resize_2d_array(cor_im_data[i][j][k]))
                dice_plot.append(d_s)
                temp_2.append(d_s)
                if d_s > val:
                    temp.append(data_possibilities[1])
                if d_s <= val:
                    temp.append(data_possibilities[0])
            temp_a.append(temp)
            temp_b.append(temp_2)
        Dice_Scores.append(temp_a)
        dice_score_values.append(temp_b)
    #print(len(Dice_Scores),len(Dice_Scores[0]),len(Dice_Scores[0][0]))
    #print(len(dice_score_values),len(dice_score_values[0]),len(dice_score_values[0][0]))
    # Creates a 16x6xz array with the dice scores atm
    if not os.path.exists(cwd + dice_score_save_folder_location + augmented):
        os.makedirs(cwd + dice_score_save_folder_location + augmented)
    np.save(cwd + dice_score_save_loc + augmented + "\\Dice_Scores",Dice_Scores)
    np.save(cwd + dice_score_save_loc + augmented + "\\Dice_Score_Values",dice_score_values)
    np.save(cwd + dice_score_save_loc + augmented + "\\Dice_Plot",dice_plot)
    return Dice_Scores, dice_score_values, dice_plot

###---------GET LOCATION OF DATA--------------###
def get_image_data(Type, data_folder, ans_augmented, ans_feat_vec):
    filename = 'init.npy'
    if ans_feat_vec == 'y' and data_folder == 'auto_feat':
        Type = Type + '_wMRI'
        filename = 'br12con18.npy'
    filepath_baseline = "\Data\Baseline scans"
    filepath_week12 = "\Data\Week 12 scans"
    filepaths = []
    for scan in scans_to_look_at:    
        filepaths.append(cwd+filepath_baseline+'\\scan'+str(scan)+'\\'+data_folder+Type+'\\')
        filepaths.append(cwd+filepath_week12+'\\scan'+str(scan)+'\\'+data_folder+Type+'\\')    
    data = []
    for i in tqdm(range(0, len(filepaths))): # should be 16 iterations
        if ans_augmented == 'y':
            temp = os.listdir(filepaths[i])
            temp_data = []
            for j in temp: # should be 6 iterations
                temp_data.append(np.load(filepaths[i]+'\\'+j).astype(np.float16))
            data.append(temp_data)
        else:
            temp_data = []
            temp_data.append(np.load(filepaths[i]+'\\'+filename).astype(np.float16))
            data.append(temp_data)
    #print(len(data),len(data[0]),len(data[0][0]),len(data[0][0][0]))
    # Data should now be a 16x6xzx512x512 array
    return data

def decompose(x,data):
    pca = decomposition.PCA(n_components=x).fit_transform(data)
    return pca

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

def resize_2d_array(data):
    temp = []
    for i in range(0,len(data)):
        for j in range(0,len(data[i])):
            temp.append(data[i][j])
    return temp

def resize_for_svm(data):
    # Originally input a 8xzx512 array (or 4xzx512)
    # Now a 8x6xzx512 array with the augmented data so add an extra for loop
    temp = []
    for i in range(0,len(data)):
        for j in range(0,len(data[i])):
            for k in range (0, len(data[i][j])):
                temp.append(data[i][j][k])
    # Output a 1D array of length 8*6*z*512 (hopefully?)
    return temp

def load_saved_dice_scores(location,ans_augmented):
    augmented = ''
    if ans_augmented == 'y':
        augmented = "_augmented"
    list_of_files = os.listdir(cwd + location + augmented)
    temp = []
    for file in list_of_files:
        temp.append(np.load(cwd + location + augmented + "\\"+ file,allow_pickle=True))
    return temp[1],temp[2],temp[0] #dice_scores, dice_score_values, dice_plot

def write_results_to_file(Results,av,folder,hyperparameter,mean,confusion_matrices,metric_for_validation):
    if SVM == 1:
        my_results_file = open(folder + '\\results.txt','w')
        my_results_file.write('Mean of dice score of data set: '+str(mean)+'\n')
        my_results_file.write('Metric optimised in validation stage: '+metric_for_validation+'\n') 
        for i in range(0,len(confusion_matrices)):
            my_results_file.write(str(confusion_matrices[i])+'\n')
        for i in range(0,len(Results)):
            my_results_file.write('Model '+str(i)+', hyperparameter(s): '+str(hyperparameter[i][0])+', '+str(hyperparameter[i][1])+'\n')
            for j in range(0,len(Results[0])):
                my_results_file.write(Results[i][j][0]+': '+str(round(Results[i][j][1],3))+u"\u00B1"+str(round(Results[i][j][2],3))+'\n')
            my_results_file.write('\n')
        my_results_file.write('Averages:\n')
        for i in range(0,len(av)):
            my_results_file.write(av[i][0]+': '+str(round(av[i][1],3))+u"\u00B1"+str(round(av[i][2],3))+'\n')
    else:
        my_results_file = open(folder + '\\results.txt','w')
        my_results_file.write('Mean of dice score of data set: '+str(mean)+'\n')
        my_results_file.write('Metric optimised in validation stage: '+metric_for_validation+'\n') 
        for i in range(0,len(Results)):
            my_results_file.write('Model '+str(i)+', hyperparameter(s): '+str(hyperparameter[i][0])+', '+str(hyperparameter[i][1])+'\n')
            for j in range(0,len(Results[0])):
                my_results_file.write(Results[i][j][0]+': '+str(Results[i][j][1])+'\n')
            my_results_file.write('\n')
        my_results_file.write('Averages:\n')
        for i in range(0,len(av)):
            my_results_file.write(av[i][0]+': '+str(av[i][1])+'\n')
    my_results_file.close()

######------------PLOTTING HISTOGRAM OF DICE SCORES------------###

def plot_dice_scores(scores_plot):
    plt.hist(scores_plot,bins=20)
    plt.ylabel('Number of scores')
    plt.xlabel('Dice score')
    plt.title('Distribution of Dice scores')
    plt.show()
    scores = np.array(scores_plot)
    return scores.mean(),np.median(scores)

###-------------------------------------------------------------------------###
def get_classifier(classifier_type,c,g,a):
    if classifier_type == 'linear_SVM':
        classifier = svm.LinearSVC(C=c,max_iter=svm_iterations, dual = False)
    if classifier_type == 'poly_SVM':
        classifier = svm.SVC(C=c,kernel='poly',degree=poly_degree,gamma=g)
    if classifier_type == 'rbf_SVM':
        classifier = svm.SVC(C=c,kernel='rbf',gamma=g)
    if classifier_type == 'sigmoid_SVM':
        classifier = svm.SVC(C=c,kernel='sigmoid',gamma=g)
    if classifier_type == 'linear_regression_ridge':
        classifier = lm.Ridge(alpha=a)
    return classifier

def validation_stage(c,g,a,classifier,Train_new,Dice_new_Train,Train_dice_score_numerical_value,Validation_new,Dice_new_Validation,Validation_dice_score_numerical_value, metric_for_validation, k, folder_temp,scores,c_vals,g_vals,a_vals,c_best,g_best,a_best,best_score):            
    if SVM == 1:
        classifier.fit(Train_new,Dice_new_Train) 
        validate = classifier.predict(Validation_new)
        score = get_metric(validate, Dice_new_Validation, Validation_dice_score_numerical_value, beta, data_possibilities, 0, metric_for_validation, k, folder_temp)
        c_vals.append(c)
        g_vals.append(g)
        a_vals.append(a)
        scores.append(score)
        if score > best_score:
            best_score = score
            c_best = c
            g_best = g
            a_best = a    
    else:
        classifier.fit(Train_new,Train_dice_score_numerical_value)
        validate = classifier.predict(Validation_new)
        score = get_metric(validate, Dice_new_Validation, Validation_dice_score_numerical_value, beta, data_possibilities, 0, metric_for_validation, k, folder_temp)
        c_vals.append(c)
        g_vals.append(g)
        a_vals.append(a)
        scores.append(score)
        if score < best_score:
            best_score = score
            c_best = c
            g_best = g
            a_best = a        
    return c_best, g_best, a_best, best_score

def train_classifier(classifier_type,Train_new, Dice_new_Train, Train_dice_score_numerical_value, Validation_new, Dice_new_Validation, Validation_dice_score_numerical_value, metric_for_validation, k, folder_temp):
    best_score = 0
    if SVM == 0:
        best_score = 1e10
    c_best = 0
    g_best = 0
    a_best = 0
    scores = []
    c_vals = []
    g_vals = []
    a_vals = []
    for i in range(range_of_validation_iterations[0],range_of_validation_iterations[1]):                
        c = (10**(i))
        a = (10**(i))
        if classifier_type == 'linear_SVM':
            g = None
            classifier = get_classifier(classifier_type, c, g, a)
            c_best, g_best, a_best, best_score = validation_stage(c,g,a,classifier,Train_new,Dice_new_Train,Train_dice_score_numerical_value,Validation_new,Dice_new_Validation,Validation_dice_score_numerical_value, metric_for_validation, k, folder_temp,scores,c_vals,g_vals,a_vals,c_best,g_best,a_best,best_score)
        elif classifier_type == 'linear_regression_ridge':
            g = None
            classifier = get_classifier(classifier_type, c, g, a)
            c_best, g_best, a_best, best_score = validation_stage(c,g,a,classifier,Train_new,Dice_new_Train,Train_dice_score_numerical_value,Validation_new,Dice_new_Validation,Validation_dice_score_numerical_value, metric_for_validation, k, folder_temp,scores,c_vals,g_vals,a_vals,c_best,g_best,a_best,best_score)           
        else:
            for x in range(range_of_validation_iterations[0],range_of_validation_iterations[1]):  
                g=(10**(x))
                classifier = get_classifier(classifier_type, c, g, a)
                c_best, g_best, a_best, best_score = validation_stage(c,g,a,classifier,Train_new,Dice_new_Train,Train_dice_score_numerical_value,Validation_new,Dice_new_Validation,Validation_dice_score_numerical_value, metric_for_validation, k, folder_temp,scores,c_vals,g_vals,a_vals,c_best,g_best,a_best,best_score)                 
    return scores,c_vals,c_best,g_vals,g_best,a_vals,a_best
    
def main():
    ans_augmented = input("Use augmented (flipped/rotated) data (y/n) -> ")   
    ans_feat_vec = input("Use MRI feature vector (y/n) ->")
    
    print("Retrieving data...")
    image_data = get_image_data(data_type, 'auto',ans_augmented, ans_feat_vec)
    Corrected_image_data = get_image_data(data_type, 'corrected',ans_augmented, ans_feat_vec)
    patient_data = get_image_data(data_type, 'auto_feat',ans_augmented, ans_feat_vec)    
    
    ans = input("Re-calculate dice scores (y/n) -> ")
    if ans =="y":
        dice_scores, dice_score_values, dice_plot = get_Dice_Score(val, patient_data, image_data, Corrected_image_data,dice_score_save_folder_location,data_possibilities, ans_augmented)
    else:
        dice_scores, dice_score_values, dice_plot = load_saved_dice_scores(dice_score_save_folder_location,ans_augmented)
    
   # We now have arrays containing the auto, corrected, and dice score arrays
   # These should now all have an extra dimension due to augmented images
   # This will be 16x6xz (note 6 may vary if we augment images more)
       # Apart from dice_plot which should 
    
    mean,median = plot_dice_scores(dice_plot)
    print(mean,median)
    ###---------------------------------------------------------###
    
    for metric_for_validation in metrics_for_validation:    #loop over all chosen metrics to be maximised in validation
        print('Maximising ',metric_for_validation,' in validation stage')
        folder_temp = folder + '\\' + metric_for_validation
        Results = []
        hyperparameters = []
        confusion_matrices = [] 
        averages = [[metric,[],[]] for metric in metrics]
        calculated_averages = [[metric,[],[]] for metric in metrics]
        Scores = []
        C_Vals = []
        G_Vals = []
        A_Vals = []
        
        #K-FOLD VALIDATION LOOP         
        for k in tqdm(range(0,k_max)):
            
            Train,Validation,Test = split_data(patient_data,k,n)
            Train_image_data,Validation_image_data,Test_image_data = split_data(image_data,k,n)
            Train_image_data_corr,Validation_image_data_corr,Test_image_data_corr = split_data(Corrected_image_data,k,n)
            Train_dice_Scores,Validation_dice_Scores,Test_dice_Scores = split_data(dice_scores,k,n)
            Train_dice_score_numerical_value,Validation_dice_score_numerical_value,Test_dice_score_numerical_value = split_data(dice_score_values,k,n)
            
            # Splitting should be alright with the augmented data
            # Training data should now be 8x6xzx512 array for example
            
            ###----------RESIZE THE ARRAYS TO BE USED IN THE SVM (2D to 1D array)------------------###
            
            # Resizing needs to be altered for the augmented images
            
            Train_new = resize_for_svm(Train)
            Validation_new = resize_for_svm(Validation)
            Test_new = resize_for_svm(Test)
            
            #Train_new = decompose(new_dimensionality,Train_new)
            #Validation_new = decompose(new_dimensionality,Validation_new)
            #Test_new = decompose(new_dimensionality,Test_new)
            
            Dice_new_Train = resize_for_svm(Train_dice_Scores)
            Dice_new_Validation = resize_for_svm(Validation_dice_Scores)
            Dice_new_Test = resize_for_svm(Test_dice_Scores)

            Train_dice_score_numerical_value = resize_for_svm(Train_dice_score_numerical_value)
            Validation_dice_score_numerical_value = resize_for_svm(Validation_dice_score_numerical_value)
            Test_dice_score_numerical_value = resize_for_svm(Test_dice_score_numerical_value)
            
            Train_new,Dice_new_Train = randomiser(Train_new,Dice_new_Train) #randomise training set 
            
            ###------------------------VALIDATION-------------------------------------------###
            
            scores,c_vals,c_best,g_vals,g_best,a_vals,a_best = train_classifier(classifier_type,Train_new, Dice_new_Train, Train_dice_score_numerical_value, Validation_new, Dice_new_Validation, Validation_dice_score_numerical_value, metric_for_validation, k, folder_temp)
            
            ###--------------------TEST BEST MODEL--------------------------###
            
            classifier = get_classifier(classifier_type, c_best, g_best, a_best)
            if SVM == 1:
                classifier.fit(Train_new,Dice_new_Train) 
            else:
                classifier.fit(Train_new,Train_dice_score_numerical_value)
                
                Train_results = classifier.predict(Train_new)
                Test_results = classifier.predict(Test_new)
                plt.scatter(Train_dice_score_numerical_value,Train_results,label='k='+str(k))
                plt.legend(loc='upper left')
                plt.title('Train data')
                plt.ylabel('Predicted dice score')
                plt.xlabel('Actual Dice Score')
                plt.savefig(folder_temp + '\\k='+str(k)+'_Training_data_regression')
                plt.show()
                train_mse = sklearn.metrics.mean_squared_error(Train_dice_score_numerical_value, Train_results)
                plt.scatter(Test_dice_score_numerical_value,Test_results,label='k='+str(k))
                plt.legend(loc='upper left')
                plt.title('Test data')
                plt.ylabel('Predicted dice score')
                plt.xlabel('Actual Dice Score')
                plt.savefig(folder_temp + '\\k='+str(k)+'_Testing_data_regression')
                plt.show()
                
            results = classifier.predict(Test_new)
            temp = []
            for metric in metrics:
                temp.append(get_metrics_and_errors(results, Test_new, Dice_new_Test, Test_dice_score_numerical_value, beta, data_possibilities, metric, classifier, k, folder_temp))
                if SVM == 0:
                    temp.append(['Training set mean squared error',train_mse,0])
            Results.append(temp)
            hyperparameters.append([c_best,g_best,a_best])
            if SVM == 1:
                confusion_matrices.append(get_confusion_matrix_data(results, Dice_new_Test, data_possibilities))
            Scores.append(scores)
            C_Vals.append(c_vals)
            G_Vals.append(g_vals)
            A_Vals.append(a_vals)
        
        #Plotting Hyper-parameters
        if SVM == 1:
            for i in range(0,len(Scores)):                
                plt.scatter(C_Vals[i],Scores[i],label='k='+str(i))
                plt.xlabel('C')
                plt.ylabel(metric_for_validation)
                plt.xscale('log')
                plt.yscale('linear')
            plt.savefig(folder_temp + '\\C')
            plt.show() 
        else:
            for i in range(0,len(Scores)):                
                plt.scatter(A_Vals[i],Scores[i],label='k='+str(i))
                plt.xlabel('alpha')
                plt.ylabel(metric_for_validation)
                plt.xscale('log')
                plt.yscale('linear')
            plt.savefig(folder_temp + '\\alpha')
            plt.show()        
        if classifier_type != 'linear_SVM' and SVM == 1:
            for i in range(0,len(Scores)):                
                plt.scatter(G_Vals[i],Scores[i],label='k='+str(i))
                plt.xlabel('gamma')
                plt.ylabel(metric_for_validation)
                plt.xscale('log')
                plt.yscale('linear')
            plt.savefig(folder_temp + '\\gamma')               
            plt.show() 
            for i in range(0,len(Scores)):                
                cm = plt.cm.get_cmap('RdYlBu')
                sc = plt.scatter(G_Vals[i],C_Vals[i],c=Scores[i],cmap=cm)
                plt.colorbar(sc)
                plt.title('Model '+str(i+1))
                plt.xlabel('gamma')
                plt.ylabel('c')
                plt.xscale('log')
                plt.yscale('log')
                plt.savefig(folder_temp + '\\k='+str(i)+' c vs gamma')
                plt.show()
            
        for i in range(0,len(Results)):
            for j in range(0,len(Results[i])):
                for k in range(0,len(averages)):
                    if Results[i][j][0] == averages[k][0]:
                        averages[k][1].append(Results[i][j][1])
                        averages[k][2].append(Results[i][j][2])
        for i in range(0,len(averages)):
            if SVM == 1:
                calculated_averages[i][1] = weighted_mean(averages[i][1], averages[i][2])
                calculated_averages[i][2] = standard_error_on_mean(averages[i][2])
            else:
                calculated_averages[i][1] = np.mean(averages[i][1])
        write_results_to_file(Results, calculated_averages, folder_temp, hyperparameters, mean, confusion_matrices, metric_for_validation)
    os.rename(folder,folder+' COMPLETE')    
    
###-------------------------DEFINE VARIABLES--------------------------------###
classifier_types = ['linear_SVM']#['linear_SVM','poly_SVM','linear_regression_ridge]
data_types = ['_reduced'] #['', '_merged', '_reduced' ,'_merged_reduced']
for classifier_type in classifier_types:
    for data_type in data_types:
        
        #-----     VARIABLES LIKELY TO BE CHANGED      -----#
        SVM = 1 #1 if SVM used, 0 if regression
        merged = 0 #1 if classes 1 and 2 are merged, 0 if not
        new_dimensionality = 100 #dimensionality reduction from 512 (for feature vector)
        poly_degree = 2
        val = 0.89 #0.9620364830557224 #threshold for dice score (between 0 and 1) median 0.99 #med: 0.95
        range_of_validation_iterations = [-2,10]
        iterations_of_bootstrapping = 10000 #(10000)
        svm_iterations = 1000 #no. iterations when training svm
        dice_score_save_folder_location = "\\Results2\\Dice_Scores_macro_th" + str(val) + data_type #"\\Results\\Dice_Scores_val_mean"
        metrics = ['Accuracy','Precision','Recall','F-beta','AUC']
        metrics_for_validation = ['AUC'] #metric which the validation stage maximises
        #metrics_for_validation = ['mean_squared_error']
        #metrics = ['mean_squared_error']
        #-----                                          -----#
        
        
        scans_to_look_at = [1,3,6,7,10,12,13,19]    #numbers of patient data to use        
        now = datetime.now()
        date_and_time_now = now.strftime("%d/%m/%Y %H:%M:%S").replace(':', ';').replace('/','-')        
        cwd = os.getcwd()        
        n = 4
        k_max = 4         
        beta = 1        
        data_possibilities = [0,1] #['bad','good']
        folder = cwd + '\\Results2\\Graphs\\' + date_and_time_now + data_type + '(threshold = '+str(val)+'; classifier type = '+classifier_type+')'
        if classifier_type == 'poly':
            folder = cwd + '\\Results2\\Graphs\\' + date_and_time_now + data_type + '(threshold = '+str(val)+'; classifier type = '+classifier_type+'; degree = '+str(poly_degree)+')'
        for metric in metrics_for_validation:
            if not os.path.exists(folder + '\\' + metric):
                os.makedirs(folder + '\\' + metric)
        main()

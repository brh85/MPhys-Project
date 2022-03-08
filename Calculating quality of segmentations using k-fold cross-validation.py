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
from sklearn import decomposition
import scipy
#import torch
#import cv2
#import pydicom
#import SimpleITK as sitk
import itertools
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

def dice_from_jaccard_index(A,B):
    if merged == 0:
        JI = sklearn.metrics.jaccard_score(A,B, average = 'weighted')
    else:
        JI = sklearn.metrics.jaccard_score(A,B,average = 'binary')
    return 2*JI/(1+JI)

def get_Dice_Score(val,training_data,im_data,cor_im_data,dice_score_save_loc,data_possibilities):
    Dice_Scores = []
    dice_score_values = []
    dice_plot = []
    for i in tqdm(range(0,len(training_data))):
        temp = []
        temp_2 = []
        for j in range(0,len(training_data[i])):
            #d_s = Dice_Score(im_data[i][j], cor_im_data[i][j], data_possibilities)
            d_s = dice_from_jaccard_index(resize_for_svm(im_data[i][j]), resize_for_svm(cor_im_data[i][j]))
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
def get_image_data(Type):
    temp = os.listdir(cwd+numpy_array_save_location+Type)
    data = []
    temp = correct_ordering(temp)
    for z in temp:
        data.append(np.load(cwd+numpy_array_save_location+Type+'\\'+z))
    return data

def correct_ordering(directory_list):
    b_or_w = ['baseline_','week12_']
    temp = []
    for scan in scans_to_look_at:    
        for el in directory_list:
            if el[:len(b_or_w[0])] == b_or_w[0]:
                num = int(el[len(b_or_w[0]):-4])
                if scan == num:
                    temp.append(el)
            if el[:len(b_or_w[1])] == b_or_w[1]:
                num = int(el[len(b_or_w[1]):-4])
                if scan == num:
                    temp.append(el)
    return temp

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
        temp.append(np.load(cwd + location + "\\"+ file,allow_pickle=True))
        print(cwd + location + "\\"+ file)
    return temp[1],temp[2],temp[0] #dice_scores, dice_score_values, dice_plot

def write_results_to_file(Results,av,folder,hyperparameter,mean,confusion_matrices,metric_for_validation):
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
    my_results_file.close()

######------------PLOTTING HISTOGRAM OF DICE SCORES------------###

def plot_dice_scores(scores_plot):
    plt.hist(scores_plot,bins=20)
    plt.ylabel('number of scores')
    plt.xlabel('dice score')
    plt.show()
    scores = np.array(scores_plot)
    return scores.mean(),np.median(scores)

###-------------------------------------------------------------------------###
def get_classifier(classifier_type,c,g):
    if classifier_type == 'linear':
        classifier = svm.LinearSVC(C=c,max_iter=svm_iterations, dual = False)
    if classifier_type == 'poly':
        classifier = svm.SVC(C=c,kernel='poly',degree=poly_degree,gamma=g)
    if classifier_type == 'rbf':
        classifier = svm.SVC(C=c,kernel='rbf',gamma=g)
    if classifier_type == 'sigmoid':
        classifier = svm.SVC(C=c,kernel='sigmoid',gamma=g)
    return classifier

def validation_stage(c,g,classifier,Train_new,Dice_new_Train,Validation_new,Dice_new_Validation,Validation_dice_score_numerical_value, metric_for_validation, k, folder_temp,scores,c_vals,g_vals,c_best,g_best,best_score):            
    classifier.fit(Train_new,Dice_new_Train)   
    validate = classifier.predict(Validation_new)
    score = get_metric(validate, Dice_new_Validation, Validation_dice_score_numerical_value, beta, data_possibilities, 0, metric_for_validation, k, folder_temp)
    c_vals.append(c)
    g_vals.append(g)
    scores.append(score)
    if score > best_score:
        best_score = score
        c_best = c
        g_best = g
    return c_best, g_best, best_score

def train_classifier(classifier_type,Train_new, Dice_new_Train, Validation_new, Dice_new_Validation, Validation_dice_score_numerical_value, metric_for_validation, k, folder_temp):
    best_score = 0
    c_best = 0
    g_best = 0
    scores = []
    c_vals = []
    g_vals = []
    for i in range(range_of_validation_iterations[0],range_of_validation_iterations[1]):                
        c=(10**(i))
        if classifier_type == 'linear':
            g = None
            classifier = get_classifier(classifier_type, c, g)
            c_best, g_best, best_score = validation_stage(c,g,classifier,Train_new,Dice_new_Train,Validation_new,Dice_new_Validation,Validation_dice_score_numerical_value, metric_for_validation, k, folder_temp,scores,c_vals,g_vals,c_best,g_best,best_score)
        else:
            for x in range(range_of_validation_iterations[0],range_of_validation_iterations[1]):  
                g=(10**(x))
                classifier = get_classifier(classifier_type, c, g)
                c_best, g_best, best_score = validation_stage(c,g,classifier,Train_new,Dice_new_Train,Validation_new,Dice_new_Validation,Validation_dice_score_numerical_value, metric_for_validation, k, folder_temp,scores,c_vals,g_vals,c_best,g_best,best_score)                 
    return scores,c_vals,c_best,g_vals,g_best
    
def main():
    image_data = get_image_data('\\auto'+data_type)
    Corrected_image_data = get_image_data('\\corrected'+data_type)
    patient_data = get_image_data('\\feature_vectors'+data_type) #patient data has size (18,113,512)    

    #16 scans - 2 x 8 patients at baseline and week 12, ordered: patient 1 baseline, patient 1 week 12, patient 2 baseline, patinet 2 week 12 etc.
    #Then usual feature vector list, with 113 slices for each data set
    
    ans = input("Re-calculate dice scores (y/n) -> ")
    #ans = 'n'
    if ans =="y":
        dice_scores, dice_score_values, dice_plot = get_Dice_Score(val, patient_data, image_data, Corrected_image_data,dice_score_save_folder_location,data_possibilities)
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
        G_Vals = []
        #K-FOLD VALIDATION LOOP 
        #NOW 4-FOLD
        
        for k in tqdm(range(0,k_max)):
            
            Train,Validation,Test = split_data(patient_data,k,n)
            Train_image_data,Validation_image_data,Test_image_data = split_data(image_data,k,n)
            Train_image_data_corr,Validation_image_data_corr,Test_image_data_corr = split_data(Corrected_image_data,k,n)
            Train_dice_Scores,Validation_dice_Scores,Test_dice_Scores = split_data(dice_scores,k,n)
            Train_dice_score_numerical_value,Validation_dice_score_numerical_value,Test_dice_score_numerical_value = split_data(dice_score_values,k,n)
        
            ###----------RESIZE THE ARRAYS TO BE USED IN THE SVM (2D to 1D array)------------------###
            
            Train_new = resize_for_svm(Train)
            Validation_new = resize_for_svm(Validation)
            Test_new = resize_for_svm(Test)
            
            Train_new = decompose(new_dimensionality,Train_new)
            Validation_new = decompose(new_dimensionality,Validation_new)
            Test_new = decompose(new_dimensionality,Test_new)
            
            Dice_new_Train = resize_for_svm(Train_dice_Scores)
            Dice_new_Validation = resize_for_svm(Validation_dice_Scores)
            Dice_new_Test = resize_for_svm(Test_dice_Scores)

            Train_dice_score_numerical_value = resize_for_svm(Train_dice_score_numerical_value)
            Validation_dice_score_numerical_value = resize_for_svm(Validation_dice_score_numerical_value)
            Test_dice_score_numerical_value = resize_for_svm(Test_dice_score_numerical_value)
            
            Train_new,Dice_new_Train = randomiser(Train_new,Dice_new_Train) #randomise training set 
            
            ###------------------------VALIDATION-------------------------------------------###
            
            scores,c_vals,c_best,g_vals,g_best = train_classifier(classifier_type,Train_new, Dice_new_Train, Validation_new, Dice_new_Validation, Validation_dice_score_numerical_value, metric_for_validation, k, folder_temp)
            
            ###--------------------TEST BEST MODEL--------------------------###
            
            classifier = get_classifier(classifier_type, c_best, g_best)
            classifier.fit(Train_new,Dice_new_Train)   
            results = classifier.predict(Test_new)
            temp = []
            for metric in metrics:
                temp.append(get_metrics_and_errors(results, Test_new, Dice_new_Test, Test_dice_score_numerical_value, beta, data_possibilities, metric, classifier, k, folder_temp))
            Results.append(temp)
            hyperparameters.append([c_best,g_best])
            confusion_matrices.append(get_confusion_matrix_data(results, Dice_new_Test, data_possibilities))
            Scores.append(scores)
            C_Vals.append(c_vals)
            G_Vals.append(g_vals)
        
        #Plotting Hyper-parameters
        for i in range(0,len(Scores)):                
            plt.scatter(C_Vals[i],Scores[i],label='k='+str(i))
            plt.xlabel('C')
            plt.ylabel(metric_for_validation)
            plt.xscale('log')
            plt.yscale('linear')
        plt.show() 
        if classifier_type != 'linear':
            for i in range(0,len(Scores)):                
                plt.scatter(G_Vals[i],Scores[i],label='k='+str(i))
                plt.xlabel('gamma')
                plt.ylabel(metric_for_validation)
                plt.xscale('log')
                plt.yscale('linear')
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
    os.rename(folder,folder+' COMPLETE')    
    
###-------------------------DEFINE VARIABLES--------------------------------###
classifier_types = ['linear']#['linear','poly']
data_types = ['_reduced'] #['', '_merged', '_reduced' ,'merged_and_reduced']
for classifier_type in classifier_types:
    for data_type in data_types:
        merged = 0 #1 if classes 1 and 2 are merged, 0 if not
        new_dimensionality = 100 #dimensionality reduction from 512 (for feature vector)
        poly_degree = 2
        now = datetime.now()
        date_and_time_now = now.strftime("%d/%m/%Y %H:%M:%S").replace(':', ';').replace('/','-')
        scans_to_look_at = [1,3,6,7,10,12,13,19]    #numbers of patient data to use
        cwd = os.getcwd()
        val = 0.95 #0.9620364830557224 #threshold for dice score (between 0 and 1) median 0.99 #med: 0.95
        range_of_validation_iterations = [-1,10]
        n = 4
        k_max = 4
        beta = 1
        iterations_of_bootstrapping = 10000 #(10000)
        data_possibilities = [0,1] #['bad','good']
        svm_iterations = 1000 #no. iterations when training svm
        numpy_array_save_location = '\\input_numpy_arrays'
        dice_score_save_folder_location = "\\Results2\\Dice_Scores_macro" #"\\Results\\Dice_Scores_val_mean"
        metrics = ['Accuracy','Precision','Recall','F-beta','AUC']
        metrics_for_validation = ['AUC'] #metric which the validation stage maximises
        folder = cwd + '\\Results2\\Graphs\\' + date_and_time_now + data_type + '(threshold = '+str(val)+'; classifier type = '+classifier_type+')'
        if classifier_type == 'poly':
            folder = cwd + '\\Results2\\Graphs\\' + date_and_time_now + data_type + '(threshold = '+str(val)+'; classifier type = '+classifier_type+'; degree = '+str(poly_degree)+')'
        for metric in metrics_for_validation:
            if not os.path.exists(folder + '\\' + metric):
                os.makedirs(folder + '\\' + metric)
        if not os.path.exists(cwd + dice_score_save_folder_location):
            os.makedirs(cwd + dice_score_save_folder_location)
        main()

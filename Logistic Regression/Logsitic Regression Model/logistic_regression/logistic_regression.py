# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys
import random as rd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#insert an all-one column as the first column
def addAllOneColumn(matrix):
    n = matrix.shape[0] #total of data points
    p = matrix.shape[1] #total number of attributes
    
    newMatrix = np.zeros((n,p+1))
    newMatrix[:,0] = np.ones(n)
    newMatrix[:,1:] = matrix

    
    return newMatrix
    
# Reads the data from CSV files, converts it into Dataframe and returns x and y dataframes
def getDataframe(filePath, filterID = 2):
    dataframe = pd.read_csv(filePath)
    
    # dataframe = dataframe.query('Category == "Adventure"')
    rating_avg = dataframe["Rating"].mean()
    # print(rating_avg)
    dataframe['Rating'] = dataframe['Rating'].fillna(rating_avg)
    dataframe = dataframe.fillna(0)
    y = dataframe['Rating']
    
    #all categories ['App Name', 'App Id', 'Category', 'Rating', 'Rating Count' , 'Installs' , 'Minimum Installs', 'Maximum Installs', 'Free','Price', 'Currency' , 'Size', 'Minimum Android', 'Developer Id', 'Developer Website', 'Developer Email', 'Released', 'Last Updated' , 'Content Rating','Privacy Policy', 'Ad Supported' ,'In App Purchases', 'Editors Choice' ,'Scraped Time']
    todrop = ['App Name', 'App Id', 'Category', 'Rating', 'Rating Count' , 'Installs' , 'Minimum Installs', 'Free', 'Currency', 'Size', 'Minimum Android', 'Developer Id', 'Developer Website', 'Developer Email', 'Released', 'Last Updated', 'Privacy Policy' ,'Scraped Time']
    # 'Ad Supported' , 'In App Purchases', 'Editors Choice'
    # todrop.pop(filterID)
    x = dataframe.drop(todrop, axis=1)
    x["Ad Supported"] = x['Ad Supported'].astype(int)
    x['In App Purchases'] = x['In App Purchases'].astype(int)
    x['Editors Choice'] = x['Editors Choice'].astype(int)
    x = pd.get_dummies(x , columns = ['Content Rating'])
    # print(x)

    return x, y

# sigmoid function
def sigmoid(z):
    # print(z)
    return 1 / (1 + np.exp(-z))  

# compute average logL
def compute_avglogL(X,y,beta):
    eps = 1e-50
    n = y.shape[0]
    avglogL = 0
    #========================#
    # STRART YOUR CODE HERE  #
    #========================#
    for i in range(n):
        # avglogL = avglogL + (y[i] * sigmoid((np.transpose(X[i]) @ beta)) - np.log((1 + np.exp(np.transpose(X[i]) @ beta))))
        siggy = sigmoid(np.transpose(X[i]) @ beta)
        # print(np.transpose(X[i]) @ beta)
        if(siggy >= 1):
            siggy = 0.999999999
        if(siggy < 0):
            print("WHAT")
        avglogL = avglogL + y[i] * np.log(siggy) + (1 - y[i]) * np.log(1 - siggy)
    avglogL = avglogL / n
    # print(n)
    avglogL = - avglogL
    #========================#
    #   END YOUR CODE HERE   #
    #========================# 
    return avglogL
    

# train_x and train_y are numpy arrays
# lr (learning rate) is a scalar
# function returns value of beta calculated using (0) batch gradient descent
def getBeta_BatchGradient(train_x, train_y, lr, num_iter, verbose):
    beta = np.random.rand(train_x.shape[1])

    p = train_x.shape[1] #total number of attributes

    
    beta = np.random.rand(p)
   
    #update beta interatively
    for iter in range(0, num_iter):
        #========================#
        # STRART YOUR CODE HERE  #
        #========================#
        #print(iter)
        for i in range(train_x.shape[0]):
            siggy = sigmoid(np.transpose(beta) @ train_x[i])
            beta = beta + lr * train_x[i].dot(train_y[i] - siggy)
            # print(beta)
            # beta = beta + lr * (train_y[iter] - (1/ (1 + np.exp(np.transpose(beta) @ train_x[iter]))))
            #========================#
            #   END YOUR CODE HERE   #
            #========================# 
        if(verbose == True and iter % 100 == 0):
            avgLogL = compute_avglogL(train_x, train_y, beta)
            print(f'average logL for iteration {iter}: {avgLogL} \t')
    return beta
    
# train_x and train_y are numpy arrays
# function returns value of beta calculated using (1) Newton-Raphson method
def getBeta_Newton(train_x, train_y, num_iter, verbose):
    p = train_x.shape[1] #total number of attributes
    
    beta = np.random.rand(p)
    ########## Please Fill Missing Lines Here ##########
    for iter in range(0, num_iter):
        #========================#
        # STRART YOUR CODE HERE  #
        #========================#
        siggy = sigmoid(train_x @ beta)
        pb1pb = siggy * (1 - siggy)

        seconderivie = - np.transpose(train_x) @ np.diag(pb1pb) @ train_x

        firstderivie = np.transpose(train_x) @ (train_y - siggy)

        beta = beta - (np.linalg.inv(seconderivie) @ firstderivie)
        #========================#
        #   END YOUR CODE HERE   #
        #========================# 
        if (verbose == True and iter % 10 == 0):
            avgLogL = compute_avglogL(train_x, train_y, beta)
            print(f'average logL for iteration {iter}: {avgLogL} \t')
    return beta
    
# def getSklearnLogistic(train_x, train_y, num_iter, verbose):
#     clf = LogisticRegression(random_state=0).fit(train_x, train_y)

    
# Linear Regression implementation
class LogisticRegression(object):
    # Initializes by reading data, setting hyper-parameters, and forming linear model
    # Forms a linear model (learns the parameter) according to type of beta (0 -  batch gradient, 1 - Newton-Raphson)
    # Performs z-score normalization if isNormalized is 1
    # Print intermidate training loss if verbose = True
    def __init__(self,lr=0.00000000005, num_iter=100, verbose = True):
        self.lr = lr
        self.num_iter = num_iter
        self.verbose = verbose
        self.train_x = pd.DataFrame() 
        self.train_y = pd.DataFrame()
        self.test_x = pd.DataFrame()
        self.test_y = pd.DataFrame()
        # self.algType = 0
        self.isNormalized = 0
       

    def load_data(self, all_data):
        x, y = getDataframe(all_data)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(x, y)
        # print(self.train_x.shape)
        # print(self.test_x.shape)
        # self.train_x, self.train_y = getDataframe(all_data)
        # self.test_x, self.test_y = getDataframe(all_data)
        
    def normalize(self):
        # Applies z-score normalization to the dataframe and returns a normalized dataframe
        self.isNormalized = 1
        data = np.append(self.train_x, self.test_x, axis = 0)
        means = data.mean(0)
        std = data.std(0)
        self.train_x = (self.train_x - means).div(std)
        self.test_x = (self.test_x - means).div(std)
    
    # Gets the beta according to input
    def train(self):
        # self.algType = algType
        newTrain_x = addAllOneColumn(self.train_x.values) #insert an all-one column as the first column
        
        beta = getBeta_BatchGradient(newTrain_x, self.train_y.values, self.lr, self.num_iter, self.verbose)
        #print('Beta: ', beta)
            
        train_avglogL = compute_avglogL(newTrain_x, self.train_y.values, beta)
        print('Training avgLogL: ', train_avglogL)
        
        return beta
            
    # Predicts the y values of all test points
    # Outputs the predicted y values to the text file named "logistic-regression-output_algType_isNormalized" inside "output" folder
    # Computes accuracy
    def predict(self, x, beta):
        newTest_x = addAllOneColumn(x)
        self.predicted_y = (sigmoid(newTest_x.dot(beta))>=0.5)
        return self.predicted_y
       
    # predicted_y and y are the predicted and actual y values respectively as numpy arrays
    # function prints the accuracy
    def compute_accuracy(self,predicted_y, y):
        acc = np.sum(predicted_y == y)/predicted_y.shape[0]
        for x,z in zip(predicted_y, y):
            # if z == 5.0:
            #     print(x, "<- pred     |     y->", z)
            if x != z:
                # if z == 0.0:
                print(x, "<- pred     |     y->", z)
        return acc
 
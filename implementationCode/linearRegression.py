import pandas as pd
import numpy as np
import random as rd
import sklearn

from sklearn.linear_model import LinearRegression as LinReg
from numpy.random import RandomState


def addAllOneColumn(matrix):
    n = matrix.shape[0]
    p = matrix.shape[1]

    newMatrix = np.zeros((n, p+1))
    newMatrix[:,1:] = matrix
    newMatrix[:,0] = np.ones(n)

    return newMatrix



def cleanData(dataframe):
    # filter out the unncessary columns
    noNoCols = ['App Name', 'App Id', 'Installs', 'Minimum Installs', 'Developer Website', 'Currency', 'Developer Id', 'Minimum Android', 'Developer Email', 'Privacy Policy', 'Scraped Time', 'Released', 'Last Updated', 'Free', 'Size']
    dataframe = dataframe.drop(noNoCols, axis = 1)

    # convert the categorical data into numerical values with one hot encoding
    oHE_data = pd.get_dummies(dataframe, columns = ['Content Rating']) 

    # set all NaN as average of the respective column
    avgRating = oHE_data['Rating'].mean()
    avgInstall = oHE_data['Maximum Installs'].mean()

    oHE_data['Rating'].fillna(avgRating, inplace=True)
    oHE_data['Maximum Installs'].fillna(avgInstall, inplace=True)

    # set all NaN as 0 for the rest of the dataset 
    oHE_data.fillna(0, inplace=True)

    # # convert the size to a float
    # oHE_data["Size"] = oHE_data["Size"].str.extract(r'(\d+)').astype(float)
    # oHE_data['Size'] = oHE_data['Size'] + 1

    # convert True/False into 0 and 1
    oHE_data["Ad Supported"] = oHE_data["Ad Supported"].astype(int)
    oHE_data["In App Purchases"] = oHE_data["In App Purchases"].astype(int)
    oHE_data["Editors Choice"] = oHE_data["Editors Choice"].astype(int)

    return oHE_data



def splitData(dataframe, categoryName):
    # take a subsection of the data with just a predetermined category
    dataframeSplit = dataframe[dataframe.Category == categoryName]

    return dataframeSplit



def makeDataframe(filepath, type, categoryName):
    # make a full dataframe with everything provided in the file
    dataframe = pd.read_csv(filepath)

    # clean the data
    cleanedData = cleanData(dataframe)

    # take a subsection of the data if asked to
    if categoryName != 'All':
        split = splitData(cleanedData, categoryName)
    else: 
        split = cleanedData

    # randomly sort into testing and training data
    rng = RandomState()

    # split into test and train data (80% goes to train and 20% goes to test)
    train = split.sample(frac = 0.8, random_state = rng)
    test = split.loc[~split.index.isin(train.index)]

    # make the dataframes
    if type == 0:   # dataset to predict rating
        train_y = train['Rating']
        train_x = train.drop(['Rating', 'Category'], axis = 1)

        test_y = test['Rating']
        test_x = test.drop(['Rating', 'Category'], axis = 1)
    elif type == 1:   # dataset to predict rating
        train_y = train['Maximum Installs']
        train_x = train.drop(['Rating', 'Rating Count', 'Maximum Installs'], axis = 1)

        test_y = test['Maximum Installs']
        test_x = test.drop(['Rating', 'Rating Count', 'Maximum Installs'], axis = 1)
    else:
        print('Invalid input!\n\t0 - predict rating\n\t1 - predict number of installs')
    
    return train_x, train_y, test_x, test_y



def trainClosedForm(train_x, train_y):

    p = train_x.shape[1]   
    beta = np.dot(np.linalg.inv(np.dot(train_x.T, train_x)), np.dot(train_x.T, train_y))
    return beta



def trainBatchGradient(train_x, train_y, lr, num_iter):
    beta = np.random.rand(train_x.shape[1])

    n = train_x.shape[0] #total of data points
    p = train_x.shape[1] #total number of attributes

    
    beta = np.random.rand(p)
    #update beta interatively
    for iter in range(0, num_iter):
       deriv = np.zeros(p)
       for i in range(n):
           deriv += train_x[i, :] * (np.matmul(np.transpose(train_x[i, :]), beta) - train_y[i])
       deriv = deriv / n
       beta = beta - deriv.dot(lr)
    return beta


def trainStochastic(train_x, train_y, lr):
    n = train_x.shape[0] #total of data points
    p = train_x.shape[1] #total number of attributes
    
    beta = np.random.rand(p)

    epoch = 3
    for iter in range(epoch):
        indices = list(range(n))
        rd.shuffle(indices)
        for i in range(n):
            idx = indices[i]
            beta = beta + lr * (train_y[idx] - (train_x[idx, :].T.dot(beta))) * train_x[idx, :]

    return beta



# Linear Regression Implementation
class LinearRegression(object):

    def __init__(self, learnRate=0.0000000000001, numIter=100):
        self.learnRate = learnRate
        self.numIter = numIter
        self.train_x = pd.DataFrame()
        self.train_y = pd.DataFrame()
        self.test_x = pd.DataFrame()
        self.test_y = pd.DataFrame()
    

    def load_data(self, dataFile, categoryName, type=0):
        self.train_x, self.train_y, self.test_x, self.test_y = makeDataframe(dataFile, type, categoryName)


    def train(self):
        newTrain_x = addAllOneColumn(self.train_x.values)
       
        beta = []

        sklearnModel = LinReg()
        sklearnModel.fit(newTrain_x, self.train_y.values)

        sklearnBeta = sklearnModel.coef_

        closedFormBeta = trainClosedForm(newTrain_x, self.train_y.values)
        batchGrad = trainBatchGradient(newTrain_x, self.train_y.values, self.learnRate, self.numIter)
        # stochGrad = trainStochastic(newTrain_x, self.train_y.values, self.learnRate)
        # print('Stochastic Trained')

        beta.append(closedFormBeta)
        beta.append(batchGrad)
        beta.append(sklearnBeta)
        # beta.append(stochGrad)
        
        return beta
    
    
    def predict_rating(self, x, beta):
        test = addAllOneColumn(x)
        predicted_y = test.dot(beta)

        # ensure that the predicted rating is in [0, 5]
        predicted_mid = np.where(predicted_y < 0, 0, predicted_y)
        predicted_final = np.where(predicted_mid > 5, 5, predicted_mid)

        return predicted_final


    def computeMSE(self, predicted_y, y):
        mse = np.sum((predicted_y - y)**2)/predicted_y.shape[0]
        return mse
#!/usr/bin/env python

import numpy as np
import pandas as pd
import math


#Code to update the data training and testing
from sys import argv

#Getting the arguments directly from the command to run the script
script, KParameter, trainingFile, testingFile = argv

training_data = pd.read_csv(trainingFile, header=None)              #creating a pandas dataframe after reading the data in the file
testing_data = pd.read_csv(testingFile, header=None)

#Implementation of the KNN algorithm


#First we need tp find the distance


numberRows, numberColumns = training_data.shape
lengthRow, lengthColumn = testing_data.shape

neighborsIndex = []
labels = []

#finding the distance of each individual multidimensional point of the testing datasets to each of the point of the training datasets

def distance(a,b):

    sum = 0.00

    list = []


    for i in range(numberRows):                         #iterating through each point of the training data set
        for j in range(numberColumns - 1 ):


            if(isinstance(a.iloc[i][j], str ) == True):     #checking if the feature is a real value or if it is string
                if(a.iloc[i][j] != b[j]):                   #each the feature of the testing dataset corresponding to the feature of the training dataset is different add 1 to the distance else add 0
                    sum = sum + 1.0
                else:
                    sum = sum + 0.0
            else:
                diff = math.pow((a.iloc[i][j] - b[j]), 2)       #Computation that happens when the feature is real value DL2(a,b) = sqrt(sum(ai , bi)**2))
                sum = sum + diff

        list.append(math.sqrt(sum))
        sum = 0

    return list  #returns a list of all the distance on the point of the testing dataset to each point of the training dataset


def label(listIndex, training_data):                                    #function to find the actual predicted label associated to each testing data

    for i in range(len(listIndex)):                                     #go through each point of the data set corresponding the indexes of the distances
        label = training_data.iloc[listIndex[i]][ numberColumns-1 ]     #finding all the labels
        labels.append(label)

    return max(set(labels), key=labels.count)                          #returning the label that appears the most



testing_data[lengthColumn] = testing_data[lengthColumn-1]

a = 0

for index,row in testing_data.iterrows():                   # iterating through each individual row of the testing data set
    neighborsDistance = distance(training_data,row)         # finding all the distance from each of the point of the testing dataset to all the individual points  of the training datasets
    sortedIndex = sorted(range(len(neighborsDistance)), key=lambda k: neighborsDistance[k]) #sorting the distance and getting the indexes associated to each point of the training dataset
    for i in range(int(KParameter)):                #getting the number of points corresponding to the k parameter value
        neighborsIndex.append(sortedIndex[i])

    predict = label(neighborsIndex, training_data)     #calling the label() function to get the most recurrent label between the points selected in the training dataset
    testing_data.loc[a, lengthColumn] = predict  #appending the predicted labels to the testing dataset

    a = a + 1


testing_data.to_csv(testingFile, header=False, index=False)     #writing the pandas dataframe to a file

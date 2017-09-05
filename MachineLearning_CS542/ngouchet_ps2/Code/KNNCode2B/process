#!/usr/bin/env python

import pandas as pd
import numpy as np
#Code to update the data training and testing
from sys import argv

#Getting the arguments directly from the command to run the script
script, first, second = argv

#This code uses panda and numpy to facilitate processing data

def update(data):


	entry = [0,3,4,5,6]
	column = [1,2,7,10,13,14]				#different columns with missing data

	data = data.replace('?', np.NaN)

	for i in entry:
		data[i] = data[i].fillna(data[i].mode()[0])			#replacing missing data by its mode


	entry = [1,13]					#real-valued features
	labels = ['+', '-']
	for col in entry:
		data[col] = data[col].apply(float)		#convert strings values into float so that it can usable
		for c in labels:
			data.loc[ (data[col].isnull()) & ( data[15]==c ), col ] = data[col][data[15] == c].mean()  #finding of the real-value condition on the value of the label and updaing the missing data

	for col in column:
		data[col] = (data[col] - data[col].mean())/data[col].std()				#normalize the data using panda functions and updating the entire file

	return data


trainingData = pd.read_csv(first, header=None)						#reading the file as a panda file and saving the data into trainingData
trainingData = update(trainingData)									# Calling the function update() that does all the work
trainingData.to_csv('crx.training.processed', header=False, index=False) # saving the file


testingData = pd.read_csv(second, header=None)					#reading the file as a panda file and saving the data into trainingData
testingData = update(testingData)								# Calling the function update() that does all the work
testingData.to_csv('crx.testing.processed', header=False, index=False)		# saving the file

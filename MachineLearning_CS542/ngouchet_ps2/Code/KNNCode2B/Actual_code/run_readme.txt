Readme For the KNN Algorithm

First i am opening the different files using pandas in order to create a panda Dataframe

Function distance(a, b)

finding the distance of each individual multidimensional point of the testing datasets to each of the point of the training datasets

Steps

iterating through each point of the training data set
checking if the feature is a real value or if it is string
each the feature of the testing dataset corresponding to the feature of the training dataset is different add 1 to the distance
 else add 0
 Computation that happens when the feature is real value DL2(a,b) = sqrt(sum(ai , bi)**2))
 returns a list of all the distance on the point of the testing dataset to each point of the training dataset


HOW THE ACTUAL CODE WORKS

iterating through each individual row of the testing data set
finding all the distance from each of the point of the testing dataset to all the individual points  of the training datasets
sorting the distance and getting the indexes associated to each point of the training dataset
getting the number of points corresponding to the k parameter value
calling the label() function to get the most recurrent label between the points selected in the training dataset
appending the predicted labels to the testing dataset
writing the pandas data frame to a file


 Function label(listIndex, training_data)
 function to find the actual predicted label associated to each testing data

 Steps

 go through each point of the data set corresponding the indexes of the distances
 finding all the labels
 returning the label that appears the most

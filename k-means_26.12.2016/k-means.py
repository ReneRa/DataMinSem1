#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TODO: act like this was created earlier?
Created on Wed Dec 21 21:33:36 2016

@author: Rene, Jonathan, Marnik
"""

#import the file
import pandas
import random as ran
import time
import matplotlib.pyplot as plt
import numpy as np


filename = "Supermarket.xlsx"
# parse_cols = "W:AL" for monetary data (cols W-AL in supermarket dataset)
dataSel = pandas.read_excel(filename, parse_cols = "W:AL")
data = dataSel.values.tolist()
lines = len(data)
columns = len(data[0])

#specify k and the maximal number of iterations

centroids = []
assignedCluster = []
newCentroids = []

test = []

maxClusters = 10
maxIterations = 300

#main function, kmeans
def kmeans(data, k, centroids, assignedCluster, newCentroids):
   centroids = []
    #initialize random centroids
   centroids = initialize_cluster(data,centroids, k) 
   #print(test)
   
   i=0
   # adjust centroids by iteration, stop when finished or max iterations
   for num in range(i, maxIterations):
        assignedCluster =[]  #delete values in list
        newCentroids =[]     #delete values in list
        #TODO: assignedCluster not really necessary as param
        assignedCluster = assign_Centroid(data, centroids, assignedCluster, k)
        newCentroids = recalculate_Centroids(newCentroids, data, assignedCluster, k)
        # recognizing natural finish point
        if centroids == newCentroids: 
            break
        centroids = newCentroids
        if (i==max):
            assignedCluster = assign_Centroid(data, centroids, assignedCluster)
        #print(centroids)
        #print (assignedCluster)
   # print ("\nSSE for k = " + str(k) + ": ")
   SSE = calculateSumSquaredError(data, centroids, assignedCluster, 2)
   print (SSE)
   return SSE
   
   

#TODO: make sure the same point is not picked twice? - could be an improvement as well...
# initializes k centroids
# picks random data points as initial centroids
def initialize_cluster(data, centroids, k):
    for cluster in range(0, k):
        centroids.append(ran.choice(data))
    return centroids
    
# Use the k means plus plus method to initialize the centroids
def initialize_plus(data, k):
    centroids.append(ran.choice(data))
    
    for cluster in range(1, k):
        # Calculate distance to nearest centroid for each data point  
        minDistances = [] # Holds the distances to the nearest centroid for each data point
        totalDistance = 0        

        for datapoint in data:
            # Store distances to all clusters and pick nearest from this
            distances = [] 

            for centroid in centroids:
                distance = 0
                for i in range(0, len(datapoint)):
                    distance += abs(datapoint[i] - centroid[i])
                distances.append(pow(distance, 2))
                
            # Pick the nearest cluster for the current data point
            minDistances.append([data.index(datapoint), min(distances)])
            totalDistance += min(distances)
        
        # Fill a cummulative list, 0 to 1, with appropiate probabilities, which is used to pick the new centroid by weighted probability
        cumProbabilities = []
        previousProb = 0
        for minDistance in minDistances:
            probability = (minDistance[1]/totalDistance) * 100
            if minDistance == minDistances[0]:
                cumProbabilities.append([minDistance[0], probability])
            else:
                cumProbabilities.append([minDistance[0], previousProb + probability])
            previousProb += probability

        r = ran.uniform(0.0, 100.0)
        for cumProbability in cumProbabilities:
            if r < cumProbability[1]:
                centroids.append(data[cumProbability[0]])
                break

    return centroids
# assigns data points to the nearest centroid
def assign_Centroid(data, centroids, assignedCluster, k):
    distance = []
    j=0
    while (j<lines):
        i=0
        currentData= data[j]     
        while (i<k):

            currentCentroid = centroids[i]
            #calculate euclidean distance from one datapoint to all centroids
            distance.append(calculate_LDistance(currentData, currentCentroid, 2))
            i = i+1
        #choose the index of the smallest difference
        assignedCluster.append(distance.index(min(distance)))
        distance[:] = []
        j =j+1
    return assignedCluster
    
def calculate_LDistance (currentData, currentCentroid, lNorm):
    return pow(sum([pow(abs(currentData - currentCentroid),lNorm) for currentData, currentCentroid in zip(currentData, currentCentroid)]),(1/lNorm))
    
def recalculate_Centroids (newCentroids, data, assignedCluster, k):
    # calculate new Centroids, by using the mean of all data points
    for num in range(0,k):
        #list of indices for datapoints assigned to chosen cluster
        index = [ idx for idx, val in enumerate(assignedCluster) if val == num]
        #list of datapoints, identified by using indices
        temparray = [data[i] for i in index] 
        #creating the sum of each dimension of the selected DataPoints
        sumarray = [sum(i) for i in zip(*temparray)] 
        #Get the number of data points of each dimension to create mean
        listLength = sum(1 for x in temparray if isinstance(x,list)) 
        #create mean for each dimension 
        newCentroids.append([x / listLength if sum else 0  for x in sumarray])
    return newCentroids
    
def calculateSumSquaredError (data, centroids, assignedCentroids, lNorm):
    sumSquareDistance = 0
    counter = 0
    while (counter<lines):
        currentData= data[counter]      
        currentCentroid = centroids[assignedCentroids[counter]]
        #calculate distance from one datapoint to assigned cluster centroid
        sumSquareDistance = sumSquareDistance + pow(calculate_LDistance(currentData, currentCentroid, lNorm),2)
        counter =counter+1
    return sumSquareDistance
    
def createElbowGraph (maxClusters, data):
    SSE = []
    for k in range (1, maxClusters+1):
        #calling the function
        SSE.append(kmeans(data, k, centroids, assignedCluster, newCentroids))
        #SSE.append()

    plt.plot([numberCluster for numberCluster in range (1, maxClusters+1)], SSE); 
    plt.xlabel('#Clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Graph')
    plt.show();

    return
    
#from http://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans
def compute_bic(data, centroids, assignedCluster, k):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    #number of clusters
    m = k
    n = []
    
    for num in range(0,k):
        #list of indices for datapoints assigned to chosen cluster
        index = [ idx for idx, val in enumerate(assignedCluster) if val == num]
        #list of datapoints, identified by using indices
        temparray = [data[i] for i in index] 
        #Get the number of data points of each dimension to create mean
        listLength = sum(1 for x in temparray if isinstance(x,list))
        n.append (listLength);
    
    #size of data set
    N, d = data.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum(calculateSumSquaredError(data, centroids, assignedCluster, 2))

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)
    
    
createElbowGraph(maxClusters, data)



#startpoint to measure the runtime
startTime = time.time()
#calling the function
k=3
kmeans(data, k, centroids, assignedCluster, newCentroids)
#finish to measure the runtime
elapsedTime = time.time() - startTime
print('The runtime for clustering is: ' + str(elapsedTime))
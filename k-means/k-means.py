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
from sklearn.decomposition.pca import PCA
from scatter_plot import plotClusters

#specify k, the maximal number of iterations and the final number of dimensions

centroids = []
assignedCluster = []
newCentroids = []
dataPointsIndex = []
finalNumberOfDimensions = 2
maxClusters = 4
maxIterations = 300

filename = "Supermarket_cleaned.xlsx"
# parse_cols = "W:AL" for monetary data (cols Q-AC in supermarket dataset)
dataSel = pandas.read_excel(filename, parse_cols = "Q:AC")

lines = len(data)
columns = len(data[0])




#main function, kmeans
def kmeans(data, k, centroids, assignedCluster, newCentroids):
    #initialize random centroids
   centroids = initialize_cluster(data, centroids, k) 
   i=0
   # adjust centroids by iteration, stop when finished or max iterations
   for num in range(i, maxIterations):
        dataPointsIndex[:] = assignedCluster
        assignedCluster =[]  #delete values in list
        newCentroids =[]     #delete values in list
        #TODO: assignedCluster not really necessary as param
        assignedCluster = assign_Centroid(data, centroids, assignedCluster, k)
        newCentroids = recalculate_Centroids(newCentroids, data, assignedCluster, k)
        # recognizing natural finish point
        if centroids == newCentroids: 
            break
            return
        centroids = newCentroids
        if (i==max):
            assignedCluster = assign_Centroid(data, centroids, assignedCluster)
        #print(centroids)
        #print (assignedCluster)
   # print ("\nSSE for k = " + str(k) + ": ")
   # SSE = calculateSumSquaredError(data, centroids, assignedCluster, 2)
   # print (SSE)
   return centroids, assignedCluster, dataPointsIndex, newCentroids
   

#TODO: make sure the same point is not picked twice? - could be an improvement as well...
# initializes k centroids
# picks random data points as initial centroids
def initialize_cluster(data, centroids, k):
    for cluster in range(0, k):
        centroids.append(ran.choice(data))
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
    
def calculateSumSquaredError (data, centroids, assignedCluster, lNorm):
    sumSquareDistance = 0
    counter = 0
    while (counter<lines):
        currentData= data[counter]      
        currentCentroid = centroids[assignedCluster[counter]]
        #calculate distance from one datapoint to assigned cluster centroid
        sumSquareDistance = sumSquareDistance + pow(calculate_LDistance(currentData, currentCentroid, lNorm),2)
        counter =counter+1
    return sumSquareDistance
    
def createElbowGraph (maxClusters, data):
    SSE = []
    cluster = []
    for k in range (1, maxClusters+1):
        kMeans = kmeans(data, k, centroids, assignedCluster, newCentroids)
        SSE.append(calculateSumSquaredError (data, kMeans[0], kMeans[1], 2))
        cluster.append(k)
           
    plt.plot(cluster, SSE); 
    plt.xlabel('#Clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Graph')
    plt.show();

    return
    
def createBICGraph (maxClusters, data):
    allBIC = []
    for k in range (1, maxClusters+1):
        clusterResult = kmeans(data, k, centroids, assignedCluster, newCentroids)
        BIC = compute_BIC(data, clusterResult[0], clusterResult[1], k)
        #print ("BIC for k=" + str(k) + ": " + str(BIC))
        allBIC.append(BIC)
    plt.plot([numberCluster for numberCluster in range (1, maxClusters+1)], allBIC); 
    plt.xlabel('#Clusters')
    plt.ylabel('BIC')
    plt.title('Bayesian Information Criterion Graph')
    plt.show();

    return
    
def getMaximalBIC (maxClusters, data):
    maximalBICPosition = 1;
    clusterResult = kmeans(data, maximalBICPosition, centroids, assignedCluster, newCentroids)
    maximalBIC = compute_BIC(data, clusterResult[0], clusterResult[1], maximalBICPosition)
    for k in range (2, maxClusters+1):
        clusterResult = kmeans(data, k, centroids, assignedCluster, newCentroids)
        currentBIC = compute_BIC(data, clusterResult[0], clusterResult[1], k)
        if (currentBIC<=maximalBIC):
            break
        else:
            maximalBIC = currentBIC
            maximalBICPosition = k
    return maximalBICPosition, maximalBIC
        

    
def compute_BIC(data, centroids, assignedCluster, k):
    """
    Computes the BIC metric for a given clusters

    from http://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans
    uses library: numpy
    
    returns BIC value for one k
    """
    #number of clusters
    n = []

    #size of data set
    N = len(data)
    d = len(data[0])
    
    for num in range(0,k):
        #list of indices for datapoints assigned to chosen cluster
        index = [ idx for idx, val in enumerate(assignedCluster) if val == num]
        #list of datapoints, identified by using indices
        temparray = [data[i] for i in index] 
        #Get the number of data points of each dimension to create mean
        listLength = sum(1 for x in temparray if isinstance(x,list))
        n.append (listLength);

    SSE = calculateSumSquaredError (data, centroids, assignedCluster, 2)
    
    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - k) / d) * SSE

    const_term = 0.5 * k * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(k)]) - const_term*k*d
    
    #BIC2 = N + N * np.log(2*np.pi) + N * np.log (SSE/N) + np.log(N) * (d+1)

    return BIC
    #Adding the index to each data point and plotting the result
def plotting(dataPointsIndex, centroids):
    kmeans(data, k, centroids, assignedCluster, newCentroids)
    counter = 0
    for dataPoint in dataPointsIndex:
        data[counter].append(int(dataPoint))
        counter += 1
    #necessary to change centroid value to int
    KMeans = kmeans(data, k, centroids, assignedCluster, newCentroids)[0]
    counter = 0    
    for centroid in KMeans:
        KMeans[counter][2] = int(KMeans[counter][2])
        counter += 1          
    #print(kmeans(data, k, centroids, assignedCluster, newCentroids)[0])    
    plotClusters(KMeans, data)


createElbowGraph(maxClusters, data)
createBICGraph(maxClusters, data)



#startpoint to measure the runtime
startTime = time.time()
#calculating k
k = getMaximalBIC(maxClusters, data)[0]
#Plot the Clustering result
plotting(dataPointsIndex, centroids)
#calling the function
print ("centroids for k-means with " + str(k) + ":")
print (kmeans(data, k, centroids, assignedCluster, newCentroids)[0])
#finish to measure the runtime
elapsedTime = time.time() - startTime
print('The total runtime for clustering with ' + str(k) + ' clusters is: ' + str(elapsedTime))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 3 21:33:36 201
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

finalNumberOfDimensions = 2
maxClusters = 8
maxIterations = 300

#TODO: Clean the dataset @Rene
filename = "GadgetManiacs_Cluster.xlsx"
# parse_cols = "W:AL" for monetary data (cols Q-AC in supermarket dataset)
dataSel = pandas.read_excel(filename, parse_cols = "J:O")
data = dataSel.values.tolist()
#data = PCA(finalNumberOfDimensions).fit_transform(originalData).tolist()

lines = len(data)
columns = len(data[0])

#main function, kmeans
def kmeans(k):

    #initialize random centroids
    initialCentroids = []
    assignedCluster = []
    newCentroids = []
    i=0
    initialCentroids = initialize_cluster(k)
    # adjust centroids by iteration, stop when finished or max iterations
    for num in range(i, maxIterations):
        #TODO: how to copy elements?
        dataPointsIndex = list(assignedCluster)
        
        assignedCluster [:] = []  #delete values in list
        assignedCluster [:] = assign_Centroid(initialCentroids, k)
        newCentroids [:] = []     #delete values in list
        newCentroids [:] = recalculate_Centroids(assignedCluster, k)
        # recognizing natural finish point
        if initialCentroids == newCentroids:
            print('iterations = ' + str(num))
            break;

        if (num==maxIterations):
            print('used max iterations..')
            assignedCluster [:] = assign_Centroid(newCentroids, k)
            
        initialCentroids [:] = list(newCentroids)

    return newCentroids, assignedCluster, dataPointsIndex, newCentroids
    
#TODO: make sure the same point is not picked twice? - could be an improvement as well...
# initializes k centroids
# picks random data points as initial centroids
def initialize_cluster(k):
    localCentroids = [];
    for cluster in range(0, k):
        ch = ran.choice(data)
        localCentroids.append(ch)
    return localCentroids
    
# Use the k means plus plus method to initialize the centroids
def initialize_plus(k):
    localCentroids = [];
    localCentroids.append(ran.choice(data))
    
    for cluster in range(1, k):
        # Calculate distance to nearest centroid for each data point  
        minDistances = [] # Holds the distances to the nearest centroid for each data point
        distances = []
        totalDistance = 0        

        for datapoint in data:
            # Store distances to all clusters and pick nearest from this
            distances [:] = [] 

            for centroid in localCentroids:
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
                localCentroids.append(data[cumProbability[0]])
                break

    return localCentroids
    
# assigns data points to the nearest centroid
def assign_Centroid(localCentroids, k):
    distance = []
    localCluster = []
    j=0
    while (j<lines):
        i=0
        currentData = data[j]     
        while (i<k):

            currentCentroid = localCentroids[i]
            #calculate euclidean distance from one datapoint to all centroids
            distance.append(calculate_LDistance(currentData, currentCentroid, 2))
            #distance.append(calculate_ChebyshevDistance(currentData, currentCentroid))
            i = i+1
        #choose the index of the smallest difference
        localCluster.append(distance.index(min(distance)))
        distance [:] = []
        j =j+1
    return localCluster
    
def calculate_LDistance (currentData, currentCentroid, lNorm):
    return pow(sum([pow(abs(currentData - currentCentroid),lNorm) for currentData, currentCentroid in zip(currentData, currentCentroid)]),(1/lNorm))

def calculate_ChebyshevDistance (currentData, currentCentroid):
    return max([abs(currentData - currentCentroid) for currentData, currentCentroid in zip(currentData, currentCentroid)]) 
    
def recalculate_Centroids (assignedCluster, k):
    # calculate new Centroids, by using the mean of all data points
    localCentroids = []
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
        localCentroids.append([x / listLength if sum else 0  for x in sumarray])
    return localCentroids
    
def calculateSumSquaredError (localCentroids, assignedCluster, lNorm):
    sumSquareDistance = 0
    counter = 0
    while (counter<lines):
        currentData= data[counter]      
        currentCentroid = localCentroids[assignedCluster[counter]]
        #calculate distance from one datapoint to assigned cluster centroid
        sumSquareDistance = sumSquareDistance + pow(calculate_LDistance(currentData, currentCentroid, lNorm),2)
        #sumSquareDistance = sumSquareDistance + pow(calculate_ChebyshevDistance(currentData, currentCentroid), 2)
        counter =counter+1
    return sumSquareDistance
    
def createElbowGraph (maxClusters):
    SSE = []
    cluster = []
    for k in range (1, maxClusters+1):
        kMeans = kmeans(k)
        SSE.append(calculateSumSquaredError (kMeans[0], kMeans[1], 2))
        cluster.append(k)
           
    plt.plot(cluster, SSE); 
    plt.xlabel('#Clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Graph')
    plt.show();

    return
    
def createBICGraph (maxClusters):
    allBIC = []
    for k in range (1, maxClusters+1):
        clusterResult = kmeans(k)
        BIC = compute_BIC(clusterResult[0], clusterResult[1], k)
        #print ("BIC for k=" + str(k) + ": " + str(BIC))
        allBIC.append(BIC)
    plt.plot([numberCluster for numberCluster in range (1, maxClusters+1)], allBIC); 
    plt.xlabel('#Clusters')
    plt.ylabel('BIC')
    plt.title('Bayesian Information Criterion Graph')
    plt.show();

    return
    
def getMaximalBIC (maxClusters):
    maximalBICPosition = 1;
    clusterResult = kmeans(maximalBICPosition)
    maximalBIC = compute_BIC(clusterResult[0], clusterResult[1], maximalBICPosition)
    for k in range (2, maxClusters+1):
        clusterResult = kmeans(k)
        currentBIC = compute_BIC(clusterResult[0], clusterResult[1], k)
        if (currentBIC<=maximalBIC):
            break
        else:
            maximalBIC = currentBIC
            maximalBICPosition = k
    return maximalBICPosition, maximalBIC
        

    
def compute_BIC(centroids, assignedCluster, k):
    """
    Computes the BIC metric for a given clusters

    from http://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans
    uses library: numpy
    
    returns BIC value for one k
    """
    #number of clusters
    n = []

    #size of data set
    N = lines
    d = columns
    
    for num in range(0,k):
        #list of indices for datapoints assigned to chosen cluster
        index = [ idx for idx, val in enumerate(assignedCluster) if val == num]
        #list of datapoints, identified by using indices
        temparray = [data[i] for i in index] 
        #Get the number of data points of each dimension to create mean
        listLength = sum(1 for x in temparray if isinstance(x,list))
        n.append (listLength);

    SSE = calculateSumSquaredError (centroids, assignedCluster, 2)
    
    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - k) / d) * SSE

    const_term = 0.5 * k * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(k)]) - const_term#*k*(d+1)
    
    #BIC2 = N + N * np.log(2*np.pi) + N * np.log (SSE/N) + np.log(N) * (d+1)

    return BIC
    
def getClusterStatistics (assignedClusters, k):
    stat = [0] * k
    for i in range (0, len(assignedClusters)):
        stat[assignedClusters[i]] = stat[assignedClusters[i]] +1
    return stat

    
    ''' 
def normalizeData(data):
    lines = len(data)
    columns = len(data[0])
    
    newdata = []
    
   for i in range(0, columns):
        avg = calculate_Average(data, i)
        stddev = calculate_StandardDeviation(data, i)
        columndata = []
        for j in range (0, lines):
            columndata.append(pow(abs(data[j][i]-avg),2)/stddev)
        newdata.append(columndata)
    '''


#Create a more dynamic plotting approach @Rene
def plotting(dataPointsIndex, centroids):
    kmeans(data, k, centroids, assignedCluster, newCentroids)
    #Adding the index to each datapoint
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
    #plotting the data in external class scatter_plot
    plotClusters(KMeans, data)

#def calculate_StandardDeviation(data, i):
    
#def calculate_Average(data, column):
#    lines = len(data)    
   
#createElbowGraph(maxClusters, data)
#createBICGraph(maxClusters, data)

#startpoint to measure the runtime
startTime = time.time()
#calculating k
#k = getMaximalBIC(maxClusters, data)[0]
#Plot the Clustering result
plotting(dataPointsIndex, centroids)
#calling the function
kMeans = kmeans(data, k, centroids, assignedCluster, newCentroids)
print ("centroids for k-means with k=" + str(k) + ":")
print (kMeans[0])
stats = getClusterStatistics(kMeans[1], k)
print ("Cluster Sizes:")
print (stats)
#finish to measure the runtime
elapsedTime = time.time() - startTime
'''
for i in range (0,5):
    kMeans = kmeans(4)
    #print (kMeans[0])
    print(calculateSumSquaredError (kMeans[0], kMeans[1], 2))
    '''
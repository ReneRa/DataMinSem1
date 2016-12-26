#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 21:33:36 2016

@author: Rene, Jonathan, Marnik
"""

#import the file
import pandas
import random as ran
import math
import time

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

maxClusters = 5
maxIterations = 300

#main function, kmeans
def kmeans(data, k, centroids, assignedCluster, newCentroids):
    #initialize random centroids
   centroids = initialize_cluster(data, centroids, k) 
   i=0
   # adjust centroids by iteration, stop when finished or max iterations
   for num in range(i, maxIterations):
        assignedCluster =[]  #delete values in list
        newCentroids =[]     #delete values in list
        #TODO: assignedCluster not really necessary as param
        assignedCluster = assign_Centroid(data, centroids, assignedCluster)
        newCentroids = recalculate_Centroids(newCentroids, data, assignedCluster)
        # recognizing natural finish point
        if centroids == newCentroids: 
            break
        centroids = newCentroids
        if (i==max):
            assignedCluster = assign_Centroid(data, centroids, assignedCluster)
        #print(centroids)
        #print (assignedCluster)
   print ("\nSSE for k = " + str(k) + ": ")
   SSE = calculateSumSquaredError(data, centroids, assignedCluster, 2)
   print (SSE)
   return SSE
   

#TODO: make sure the same point is not picked twice?
# initializes k centroids
# picks random data points as initial centroids
def initialize_cluster(data, centroids, k):
    for cluster in range(0, k):
        centroids.append(ran.choice(data))
    return centroids
    
# assigns data points to the nearest centroid
def assign_Centroid(data, centroids, assignedCluster):
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
    
def recalculate_Centroids (newCentroids, data, assignedCluster):
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
    

def plotClusters(centroids, points):
    import matplotlib.pyplot as plot
    
    colors = ["b", "g", "r"]
    markers = ["o", "o", "o"]
#    markers = ["^", "s", ""]

    fig, ax = plot.subplots()
    
    index = 0
    for centroid in centroids:
        ax.scatter(centroid[0], centroid[1], color=colors[index], s=500, marker="x")
        ax.annotate("C" + str(index + 1), (centroid[0] + 2, centroid[1] + 2))
        index = (index + 1) % len(colors)
        
    index = 0
    for point in points:
        ax.scatter(point[0], point[1], color=colors[point[2] - 1], s=100, marker=markers[index])
        ax.annotate(str(point[2]), (point[0] + 1, point[1] + 1))
        index = (index + 1) % len(colors)

    fig.canvas.draw()
    fig.show()

    
#calling the function
SSE = []
for k in range (1, maxClusters+1):
    #startpoint to measure the runtime
    startTime = time.time()
    SSE.append(kmeans(data, k, centroids, assignedCluster, newCentroids))
    #finish to measure the runtime
    elapsedTime = time.time() - startTime
    print('The runtime is: ' + str(elapsedTime))
    
plotClusters([k for k in range (1, maxClusters+1)], SSE)

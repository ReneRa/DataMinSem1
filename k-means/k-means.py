#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 21:33:36 2016

@author: Rene, Jonathan, Marnik
"""

#import the file
import pandas
import random as ran

from math import *

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

k=7
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
        assignedCluster = assign_Centroid(data, centroids, assignedCluster)
        newCentroids = recalculate_Centroids(newCentroids, data, assignedCluster)
        # recognizing natural finish point
        if centroids == newCentroids: 
            break
        centroids = newCentroids
        print(centroids)
        #print (assignedCluster)
        print ("\nAvg. Error: ")
        print (calculateAverageError(data, centroids, assignedCluster, 1))
   print ("\nAvg. Error: ")
   print (calculateAverageError(data, centroids, assignedCluster, 1))
   print (calculateAverageError(data, centroids, assignedCluster, 2))
   print (calculateAverageError(data, centroids, assignedCluster, 3))
   print (calculateAverageError(data, centroids, assignedCluster, 4))
   return
   

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
    c=0 
    for num in range(c,k):
        #list of indices for datapoints assigned to chosen cluster
        index = [ h for h, l in enumerate(assignedCluster) if l == c]
        #list of datapoints, identified by using indices
        temparray = [data[i] for i in index] 
        #creating the sum of each dimension of the selected DataPoints
        sumarray = [sum(i) for i in zip(*temparray)] 
        #Get the number of data points of each dimension to create mean
        listLength = sum(1 for x in temparray if isinstance(x,list)) 
        #create mean for each dimension 
        newCentroids.append([x / listLength if sum else 0  for x in sumarray])
        c=c+1
    return newCentroids
    
def calculateAverageError (data, centroids, assignedCentroids, lNorm):
    sumDistance = 0;
    counter = 0;
    while (counter<lines):
        currentData= data[counter]      
        currentCentroid = centroids[assignedCentroids[counter]]
        #calculate distance from one datapoint to assigned cluster centroid
        sumDistance = sumDistance + calculate_LDistance(currentData, currentCentroid, lNorm)
        counter =counter+1
    return sumDistance/counter

#calling the function
kmeans(data, k, centroids, assignedCluster, newCentroids)

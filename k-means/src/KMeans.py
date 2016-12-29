import Statics
import random as ran

class kmeans():
    def __init__(self, initializationMethod):
        self.initializationMethod = initializationMethod

    def main(self, k):
        assignedCluster = []
        newCentroids = []

        # Calculate initial centroids
        if self.initializationMethod == "plus":
            initialCentroids = self.initializePlus(k)
        elif self.initializationMethod == "normal":
            initialCentroids = self.initializeClusters(k)
        elif self.initializationMethod == "spread":
            initialCentroids = self.initializeSpread(k)

        # adjust centroids by iteration, stop when finished or max iterations
        for num in range(0, Statics.maxIterations):
            #TODO: how to copy elements?
            assignedCluster [:] = []  #delete values in list
            assignedCluster [:] = self.assign_Centroid(initialCentroids, k)
            newCentroids [:] = []     #delete values in list
            newCentroids [:] = self.recalculate_Centroids(assignedCluster, k)
            # recognizing natural finish point
            if initialCentroids == newCentroids:
                print('iterations: ' + num)
                break;

            if num==Statics.maxIterations:
                assignedCluster [:] = self.assign_Centroid(newCentroids, k)

            initialCentroids [:] = list(newCentroids)

        return newCentroids, assignedCluster


    # initializes k centroids
    # picks random data points as initial centroids
    def initializeClusters(self, k):
        centroids = [];
        for cluster in range(0, k):
            ch = ran.choice(Statics.data)
            centroids.append(ch)
        return centroids

    def initializePlus(self, k):
        centroids = [];
        centroids.append(ran.choice(Statics.data))

        for cluster in range(1, k):
            # Calculate distance to nearest centroid for each data point
            minDistances = []  # Holds the distances to the nearest centroid for each data point
            distances = []
            totalDistance = 0

            for datapoint in Statics.data:
                # Store distances to all clusters and pick nearest from this
                distances [:] = []

                for centroid in centroids:
                    distances.append(self.calculate_LDistance(datapoint, centroid, 2))

                # Pick the nearest cluster for the current data point
                minDistances.append([Statics.data.index(datapoint), min(distances)])
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
                    centroids.append(Statics.data[cumProbability[0]])
                    break

        return centroids

    # Picks points as far away from each other as possible as initial clusters
    def initializeSpread(self, k):
        centroids = []

        #  Get minimum values for all dimensions and use these as first centroid
        minimumValues = []
        for dataPoint in Statics.data:
            if dataPoint == Statics.data[0]:
                for dimension in range(0, len(dataPoint)):
                    minimumValues.append(dataPoint[dimension])
            else:
                for dimension in range(0, len(dataPoint)):
                    if dataPoint[dimension] < minimumValues[dimension]:
                        minimumValues[dimension] = dataPoint[dimension]
        centroids.append(minimumValues)

        # Calculate all distances from data points to their nearest cluster
        for cluster in range(1, k):
            # Calculate distance to nearest centroid for each data point
            minDistances = []  # Holds the distances to the nearest centroid for each data point
            distances = []

            for datapoint in Statics.data:
                # Store distances to all clusters and pick nearest from this
                distances[:] = []

                for centroid in centroids:
                    distances.append(self.calculate_LDistance(datapoint, centroid, 2))

                # Pick closest cluster for the current data point
                minDistances.append(min(distances))

            # Select the data point of which its nearest cluster is the furthest away, and use this as the new centroid
            centroids.append(Statics.data[minDistances.index(max(minDistances))])

        return centroids

    def calculate_LDistance (self, currentData, currentCentroid, lNorm):
        return pow(sum([pow(abs(currentData - currentCentroid),lNorm) for currentData, currentCentroid in zip(currentData, currentCentroid)]),(1/lNorm))

    def calculate_ChebyshevDistance (self, currentData, currentCentroid):
        return max([abs(currentData - currentCentroid) for currentData, currentCentroid in zip(currentData, currentCentroid)])

    # assigns data points to the nearest centroid
    def assign_Centroid(self, centroids, k):
        distance = []
        cluster = []
        j=0
        while (j<Statics.lines):
            i=0
            currentData = Statics.data[j]
            while (i<k):

                currentCentroid = centroids[i]
                #calculate euclidean distance from one datapoint to all centroids
                distance.append(self.calculate_LDistance(currentData, currentCentroid, 2))
                #distance.append(calculate_ChebyshevDistance(currentData, currentCentroid))
                i = i+1
            #choose the index of the smallest difference
            cluster.append(distance.index(min(distance)))
            distance [:] = []
            j =j+1
        return cluster

    def recalculate_Centroids (self, assignedCluster, k):
        # calculate new Centroids, by using the mean of all data points
        localCentroids = []
        for num in range(0,k):
            #list of indices for datapoints assigned to chosen cluster
            index = [ idx for idx, val in enumerate(assignedCluster) if val == num]
            #list of datapoints, identified by using indices
            temparray = [Statics.data[i] for i in index]
            #creating the sum of each dimension of the selected DataPoints
            sumarray = [sum(i) for i in zip(*temparray)]
            #Get the number of data points of each dimension to create mean
            listLength = sum(1 for x in temparray if isinstance(x,list))
            #create mean for each dimension
            localCentroids.append([x / listLength if sum else 0  for x in sumarray])
        return localCentroids

    def calculateSumSquaredError (self, localCentroids, assignedCluster, lNorm):
        sumSquareDistance = 0
        counter = 0
        while (counter<Statics.lines):
            currentData= Statics.data[counter]
            currentCentroid = localCentroids[assignedCluster[counter]]
            #calculate distance from one datapoint to assigned cluster centr oid
            sumSquareDistance = sumSquareDistance + pow(self.calculate_LDistance(currentData, currentCentroid, lNorm),2)
            #sumSquareDistance = sumSquareDistance + pow(calculate_ChebyshevDistance(currentData, currentCentroid), 2)
            counter =counter+1
        return sumSquareDistance

    def getClusterStatistics (self, assignedClusters, k):
        stat = [0] * k
        for i in range(0, len(assignedClusters)):
            stat[assignedClusters[i]] = stat[assignedClusters[i]] +1
        return stat
import Statics

import matplotlib.pyplot as plt
import numpy as np

from scatter_plot import plotClusters

class Plot():
    def __init__(self, kMeans):
        self.kMeans = kMeans

    def createElbowGraph(self):
        SSE = []
        cluster = []
        for k in range (1, Statics.maxClusters+1):
            result = self.kMeans.main(k)
            SSE.append(self.kMeans.calculateSumSquaredError (result[0], result[1], 2))
            cluster.append(k)

        plt.plot(cluster, SSE);
        plt.xlabel('#Clusters')
        plt.ylabel('SSE')
        plt.title('Elbow Graph')
        plt.show();

        return

    def createBICGraph (self):
        allBIC = []
        for k in range (1, Statics.maxClusters+1):
            result = self.kMeans.main(k)
            BIC = self.compute_BIC(result[0], result[1], k)
            #print ("BIC for k=" + str(k) + ": " + str(BIC))
            allBIC.append(BIC)
        plt.plot([numberCluster for numberCluster in range (1, Statics.maxClusters+1)], allBIC);
        plt.xlabel('#Clusters')
        plt.ylabel('BIC')
        plt.title('Bayesian Information Criterion Graph')
        plt.show();

        return

    def compute_BIC(self, centroids, assignedCluster, k):
        """
        Computes the BIC metric for a given clusters

        from http://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans
        uses library: numpy

        returns BIC value for one k
        """
        #number of clusters
        n = []

        #size of data set
        N = Statics.lines
        d = Statics.columns

        for num in range(0,k):
            #list of indices for datapoints assigned to chosen cluster
            index = [ idx for idx, val in enumerate(assignedCluster) if val == num]
            #list of datapoints, identified by using indices
            temparray = [Statics.data[i] for i in index]
            #Get the number of data points of each dimension to create mean
            listLength = sum(1 for x in temparray if isinstance(x,list))
            n.append (listLength);

        SSE = self.kMeans.calculateSumSquaredError (centroids, assignedCluster, 2)

        #compute variance for all clusters beforehand
        cl_var = (1.0 / (N - k) / d) * SSE

        const_term = 0.5 * k * np.log(N) * (d+1)

        BIC = np.sum([n[i] * np.log(n[i]) -
                   n[i] * np.log(N) -
                 ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
                 ((n[i] - 1) * d/ 2) for i in range(k)]) - const_term#*k*(d+1)

        #BIC2 = N + N * np.log(2*np.pi) + N * np.log (SSE/N) + np.log(N) * (d+1)

        return BIC

    #Create a more dynamic plotting approach @Rene
    #TODO: Create a more dynamic plotting approach @Rene
    def plotting(self, result):

        #Adding the index to each datapoint
        counter = 0
        tempData = Statics.data
        for dataPoint in result[1]:
            tempData[counter].append(int(dataPoint))
            counter += 1

        counter = 0
        for centroid in result[0]:
            result[0][counter].append(counter)
            result[0][counter][2] = int(result[0][counter][2])
            counter += 1
        #plotting the data in external class scatter_plot
        plotClusters(result[0], tempData)
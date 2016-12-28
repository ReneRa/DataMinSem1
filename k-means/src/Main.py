import KMeans
import Plotter
import Statics

import time

createElbowGraph = False
createBICGraph = False
showClusterStats = False


def getMaximalBIC():
    maximalBICPosition = 1;
    result = kMeans.main(maximalBICPosition)
    maximalBIC = plotter.compute_BIC(result[0], result[1], maximalBICPosition)
    for k in range (2, Statics.maxClusters+1):
        clusterResult = kMeans.main(k)
        currentBIC = plotter.compute_BIC(clusterResult[0], clusterResult[1], k)
        if (currentBIC<=maximalBIC):
            break
        else:
            maximalBIC = currentBIC
            maximalBICPosition = k
    return maximalBICPosition, maximalBIC


kMeans = KMeans.kmeans(Statics.initializationMethod)
plotter = Plotter.Plot(kMeans)

k = getMaximalBIC()[0]

if createElbowGraph:
    plotter.createElbowGraph()
if createBICGraph:
    plotter.createBICGraph()


startTime = time.time()
result = kMeans.main(k)
print(kMeans.calculateSumSquaredError(result[0], result[1], 2))
elapsedTime = time.time() - startTime

print('Execution time: ' + str(elapsedTime))

if showClusterStats:
    stats = kMeans.getClusterStatistics(result[1], k)
    print(stats)

plotter.plotting(result)
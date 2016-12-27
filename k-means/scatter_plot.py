
def plotClusters(centroids, points):
    import matplotlib.pyplot as plot
    
    colors = ["b", "g", "r"]
    markers = ["o", "o", "o"]
#    markers = ["^", "s", ""]

    fig, ax = plot.subplots()
    
  
        
    index = 0
    for point in points:
        ax.scatter(point[0], point[1], color=colors[point[2] - 1], s=600, marker=markers[index])
        #ax.annotate(str(point[2]), (point[0] + 1, point[1] + 1))
        index = (index + 1) % len(colors)
    index = 0
    for centroid in centroids:
        ax.scatter(centroid[0], centroid[1], color=colors[index], s=1000, marker="x")
        #ax.annotate("C" + str(index + 1), (centroid[0] + 2, centroid[1] + 2))
        index = (index + 1) % len(colors)

    fig.canvas.draw()
    fig.show()


#if __name__ == "__main__":
#    from random import randint, random
#    
#    clusters = 3
#    numberOfPoints = 50
#    
#    # generate random centroids with coordinates between 0 and 100
#    centroids = [[random() * 100, random() * 100] for i in range(clusters)]
#    # generate random points with coordinates between 0 and 100
#    points = [[random() * 100, random() * 100, randint(1, clusters)] for i in range(numberOfPoints)]      
#    
#    plotClusters(centroids, points)
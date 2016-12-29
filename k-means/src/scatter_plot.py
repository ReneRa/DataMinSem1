#ToDo: make class more generic, works only for k=<3
def plotClusters(centroids, points):
    import matplotlib.pyplot as plot
    
    colors = ["b", "g", "r","c","m","y","k"]
    markers = ["o", "o", "o","o","o","o","o"]
#    markers = ["^", "s", ""]

    fig, ax = plot.subplots()



     
    index = 0
    for point in points:
        ax.scatter(point[0], point[1], color=colors[point[-1] - 1], s=200, marker=markers[index])
        #ax.annotate(str(point[2]), (point[0] + 1, point[1] + 1))
        index = (index + 1) % len(colors)
        plot.title('Principal Component Analysis')
        
    index = 0
    for centroid in centroids:
        ax.scatter(centroid[0], centroid[1], color=colors[index], s=1000, marker="x")
        #ax.annotate("C" + str(index + 1), (centroid[0] + 2, centroid[1] + 2))
        index = (index + 1) % len(colors)  
  

    fig.canvas.draw()
    fig.show()

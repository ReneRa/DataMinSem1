import pandas
from sklearn.decomposition.pca import PCA

finalNumberOfDimensions = 2
maxClusters = 8
maxIterations = 300
maxClusters = 5

initializationMethod = 'plus'

#TODO: Clean the dataset @Rene
filename = "../GadgetManiacs_Cleaned.xlsx"
# parse_cols = "W:AL" for monetary data (cols Q-AC in supermarket dataset)
dataSel = pandas.read_excel(filename, parse_cols="B, E:G, I:Z")
originalData = dataSel.values.tolist()
data = PCA(finalNumberOfDimensions).fit_transform(originalData).tolist()

lines = len(data)
columns = len(data[0])
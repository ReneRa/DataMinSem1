import pandas
from sklearn.decomposition.pca import PCA

finalNumberOfDimensions = 2
maxClusters = 8
maxIterations = 300
maxClusters = 10

initializationMethod = 'plus'  # normal, plus or spread

#TODO: Clean the dataset @Rene
filename = "../GadgetManiacs_Cleaned.xlsx"
# parse_cols = "W:AL" for monetary data (cols Q-AC in supermarket dataset)
dataSel = pandas.read_excel(filename, parse_cols="B, E:G, I:Z")
originalData = dataSel.values.tolist()
data = PCA(finalNumberOfDimensions).fit_transform(originalData).tolist()

lines = len(data)
columns = len(data[0])

def normalizeData():    
    for column in range(0, columns):
        avg = calculate_Average(column)
        stddev = calculate_StandardDeviation(column)
        for line in range (0, lines):
            data[line][column] = (data[line][column]-avg)/stddev
    return data

def calculate_Average(column):
     return sum([data [line][column] for line in range (0, lines)])/lines
    
def calculate_StandardDeviation(column):
     return pow(calculate_Variance(column),1/2)
     
def calculate_Variance(column):
     avg = calculate_Average(column)
     return sum([pow((data[line][column] - avg),2) for line in range (0, lines)])/lines
     
data = normalizeData()

for column in range (0, columns):
    print("avg: " + str(calculate_Average(column)))
    print("std dev. " + str(calculate_StandardDeviation(column)))
    print("min " + str(min([data[line][column] for line in range (0, lines)])))
    print("max " + str(max([data[line][column] for line in range (0, lines)])))

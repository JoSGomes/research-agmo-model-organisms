from os import walk
from scipy.io import arff
import pandas as pd

path = "..\\results"
pathDatasets = "..\\src\\main\\datasets"
all_dataframes = {}

def get_paths_folders(path, files):
    d = []
    for (dirpath, dirnames, filenames) in walk(path):
        if files:
            d.extend(filenames[0:10])
        else:
            d.extend(dirnames)
        break
    return d


def get_max_value_and_index(file):
        i = 0
        MaxIndex = 0
        MaxValue = 0.0
        for line in file.readlines():
            gmean = float(line.split(" ")[0]) * (-1)
            if gmean > MaxValue:
                MaxValue = gmean
                MaxIndex = i   
            i += 1
        return MaxValue, MaxIndex

def get_index_selected_attr(file, MaxIndex): 
    for i in range(MaxIndex):
        var.readline()
    variables = var.readline()
    j = 0
    indexTrueVariables = list()
    for variable in variables:
        if variable == "1":
            indexTrueVariables.append(j)
        j += 1
    return indexTrueVariables

def get_attrs_dataset(organism, threshold = None, thresholdNumber = None):
    if threshold:
        data = arff.loadarff('..\\src\\main\\datasets\\threshold-'+ thresholdNumber +'\\' + organism + '-threshold-' + thresholdNumber + '.arff')
        df = pd.DataFrame(data[0])
        attrFrequencies = pd.DataFrame(columns = df.columns.to_numpy()) 
        attrFrequencies.loc[0, df.columns.to_numpy()[0:]] = 0
        attrFrequencies = attrFrequencies.T.drop('class', axis='rows')
        return attrFrequencies
    else:
        data = arff.loadarff('..\\src\\main\\datasets\\' + organism + '.arff')
        df = pd.DataFrame(data[0])
        attrFrequencies = pd.DataFrame(columns = df.columns.to_numpy()) 
        attrFrequencies.loc[0, df.columns.to_numpy()[0:]] = 0
        attrFrequencies = attrFrequencies.T.drop('class', axis='rows')
        return attrFrequencies

#falta gerar 4 tabelas por threshold (fly, mouse, worm, yeast) e (fly-threshold-3, mouse-threshold-3, worm-threshold-3, yeast-threshold-3)
#dividir por classificador?
    
#attrFrequencies.iloc[2, 3] = 1
d = get_paths_folders(path, False)
for dirname in d: #\results\j48 (if not threshold)
    if dirname.__contains__("threshold"):
        thresholdNumber = dirname.split("-")[1] #number of threshold
        varPath = ""
        newPath = path + "\\" + dirname
        d1 = get_paths_folders(newPath, False)
        for dirname1 in d1: #results\threshold\j48
            newPath2 = newPath + "\\" + dirname1
            varPath2 = newPath2
            d2 = get_paths_folders(newPath2, False) 
            for dirname2 in d2:#\results\j48\FUN-fly
                if not dirname2.__contains__("VAR"):
                    organism = dirname2.split("-")[1] 
                    newPath3 = newPath2 + "\\" + dirname2
                    varPath3 = varPath2 + "\\VAR-" + organism                
                    f = get_paths_folders(newPath3, True)#\results\j48\FUN-fly\fold-0
                    for filename in f:
                        with open(newPath3 + "\\" + filename, "r") as fun:
                            MaxValue, MaxIndex = get_max_value_and_index(fun)                  
                            with open(varPath3 + "\\" + filename, "r") as var:
                                indexTrueVariables = get_index_selected_attr(var, MaxIndex)
                                if not (organism + '-threshold-' + thresholdNumber) in all_dataframes:
                                    all_dataframes.update({organism + '-threshold-' + thresholdNumber: get_attrs_dataset(organism, True, thresholdNumber)})                        
                                for index in indexTrueVariables:
                                    dataset = all_dataframes[organism + '-threshold-' + thresholdNumber]
                                    dataset.iloc[index, 0] += 1

    else:
        varPath = ""
        newPath = path + "\\" + dirname
        varPath = newPath
        d1 = get_paths_folders(newPath, False)   
        for dirname1 in d1:#\results\j48\FUN-fly
            if not dirname1.__contains__("VAR"):
                organism = dirname1.split("-")[1]
                newPath2 = newPath + "\\" + dirname1
                varPath2 = varPath + "\\VAR-" + organism
                f = get_paths_folders(newPath2, True)#\results\j48\FUN-fly\fold-0
                for filename in f:
                    with open(newPath2 + "\\" + filename, "r") as fun:
                        MaxValue, MaxIndex = get_max_value_and_index(fun)
                        with open(varPath2 + "\\" + filename, "r") as var:
                            indexTrueVariables = get_index_selected_attr(var, MaxIndex)
                            if not (organism) in all_dataframes:
                                all_dataframes.update({organism: get_attrs_dataset(organism, )})
                            for index in indexTrueVariables:
                                dataset = all_dataframes[organism]
                                dataset.iloc[index, 0] += 1                            

for key in all_dataframes:
    dataframe = all_dataframes[key]
    dataframe = dataframe.sort_values(by=0, ascending=False)
    dataframe.to_csv('tabels\\'+ key +'.csv')
    #for i in range(10):
         
        #with open(newPath+f"\\FUN-fly\\fold-{i}", "r") as file:
            #print(file.read())
from scipy import stats
from data_processing import get_paths_folders

import pandas as pd


path = "..\\results"

def slice_dataframe(data):
    UniqueNames = data.organism.unique()

    #create a data frame dictionary to store your data frames
    DataFrameDict = {elem : pd.DataFrame for elem in UniqueNames}

    for key in DataFrameDict.keys():
        DataFrameDict[key] = data[data.organism == key][:]
    
    return DataFrameDict

d = get_paths_folders(path, False)
results_classifiers = {}
for dirname in d: #\results\j48 (if not threshold)    
    if dirname.__contains__("threshold"):
        results_classifiers[dirname] = {}
        i = 0

    else:
        results_classifiers[dirname] = {}
        varPath = ""
        newPath = path + "\\" + dirname
        f = get_paths_folders(newPath, True)
        for filename in f:        
            df = pd.read_csv(newPath + "\\" + filename)
            df = df.replace(",", ".", regex=True)
            df = df.astype({"GMean": float, "ratioReduction": float})
            DataFrameDict = slice_dataframe(df)
            for key in DataFrameDict.keys():           
                k = 0
                for value in DataFrameDict[key]["GMean"]:  
                    if value < 0:
                        value = value * (-1)
                    DataFrameDict[key].at[k, "GMean"] = value
                    k = k + 1                 
                DataFrameDict[key] =  DataFrameDict[key].sort_values(by='Fold', ascending=True)

            
            for key in DataFrameDict.keys():
                if key not in results_classifiers[dirname].keys():
                    results_classifiers[dirname][key] = list()                   
                results_classifiers[dirname][key].append(DataFrameDict[key])  
                print("adicionou")

print(results_classifiers['J48']['fly'][0])


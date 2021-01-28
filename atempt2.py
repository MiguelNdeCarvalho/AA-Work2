import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

#prep
from sklearn.model_selection import cross_val_score

#models
from sklearn.ensemble import GradientBoostingClassifier

#pred libraries
#from sklearn.cross_pred import StratifiedKFold
from IPython.display import display
from sklearn import metrics



def splitPrograms(data):
    
    print(data)
    Info = data.loc[data['Program'] == "Informatics"].copy()
    print(Info)

    return Info #,Nursing,Management,Biology 

data = pd.read_csv('train.csv', header=0)
splitPrograms(data)

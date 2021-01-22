import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

#prep
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

#models
from sklearn.ensemble import GradientBoostingClassifier

#pred libraries
#from sklearn.cross_pred import StratifiedKFold
from IPython.display import display
from sklearn import metrics

def data_treatment(file):

    data = pd.read_csv(file, header=0)
    #print(file)
    #print(data)

    data.drop('Id',axis=1,inplace=True) 
   
    #obj_data = train_data.select_dtypes(include=['object']).copy()
    #print(obj_train_data['Program'].value_counts())
   
    enconde = {"Program": {"Informatics": 0, "Nursing": 1, "Management": 2, "Biology": 3}}
    data = data.replace(enconde)
    #print(train_data.head())


    #split treino em x_train e y_train
    y_data = data[['Failure']].copy()
    #print(y_data)

    data.drop('Failure',axis=1,inplace=True)
    x_data = data.copy()
    #print(x_data)

    return x_data,y_data

def best_parameters():
    return 0;
#----------------------------------main-------------------------------------#
"""
#merge sampleSubmission with test
df_submission = pd.read_csv("sampleSubmission.csv", header=0)
df_test = pd.read_csv("test.csv", header=0)

Failures = df_submission["Failure"].copy()
df_test.insert(loc=31,column ='Failure',value=Failures)
#df_test.to_csv(r'sub+test.csv', index = False)
"""

#data_treatment
x_train,y_train = data_treatment('train.csv')
x_test,y_test = data_treatment('sub+test.csv')

#split train to create pred data 


x_train, x_pred, y_train, y_pred = train_test_split(x_train, y_train, random_state=1) #default test_size=25



clf = GradientBoostingClassifier(n_estimators=2000, max_depth=1)
clf.fit(x_train, y_train.values.ravel())

final_score = clf.score(x_pred, y_pred) 
print(f"final_score: {final_score*100}")

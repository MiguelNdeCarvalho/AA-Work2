import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

#prep
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#models
from sklearn.ensemble import GradientBoostingClassifier

#validation libraries
#from sklearn.cross_validation import StratifiedKFold
from IPython.display import display
from sklearn import metrics

def data_treatment(file):

    data = pd.read_csv(file, header=0)
    print(file)
    print(data)

    data.drop('Id',axis=1,inplace=True) 
   
    #obj_data = train_data.select_dtypes(include=['object']).copy()
    #print(obj_train_data['Program'].value_counts())
   
    enconde = {"Program": {"Informatics": 0, "Nursing": 1, "Management": 2, "Biology": 3}}
    data = data.replace(enconde)
    #print(train_data.head())


    #split treino em x_train e y_train
    y_data = data[['Failure']].copy()
    print(y_data)

    data.drop('Failure',axis=1,inplace=True)
    x_data = data.copy()
    print(x_data)

    return x_data,y_data

#----------------------------------main-------------------------------------#

df_submission = pd.read_csv("sampleSubmission.csv", header=0)
df_test = pd.read_csv("test.csv", header=0)

Failures = df_submission["Failure"].copy()


#print(Failures)
df_test.insert(loc=31,column ='Failure',value=Failures)
#print(df_test)
df_test.to_csv(r'sub+test.csv', index = False)

x_train,y_train = data_treatment('train.csv')
x_test,y_test = data_treatment('sub+test.csv')


"""
data=np.genfromtxt("train.csv", delimiter=",", dtype=None, encoding=None)
x_train=data[1:,0:-1]    #  dados: da segunda à ultima linha, da primeira à penúltima coluna  
y_train=data[1:,-1]      # classe: da segunda à ultima linha, só última coluna

atributos = data[0,0:-1]

data=np.genfromtxt("test.csv", delimiter=",", dtype=None, encoding=None)
x_test=data[1:,0:-1]    #  dados: da segunda à ultima linha, da primeira à penúltima coluna  
y_test=data[1:,-1]   

#validation data 
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, random_state=1) #default test_size=25

#encode non-numerical labels
le = preprocessing.LabelEncoder()
for row in x_train
    le.fit(row)

#clf = GradientBoostingClassifier()
#clf.fit(x_train, y_train.values.ravel())


#print(f"Atributos: {atributos}\n\n")
#print(f"X_TRAIN: {x_train},\n\n Y_TRAIN: {y_train},\n\n")
#print(f"X_VALIDATION: {x_validation},\n\n Y_VALIDATION: {y_validation},\n\n")
#print(f"X_TEST {x_test},\n\n Y_TEST: {y_test}\n\n")
"""
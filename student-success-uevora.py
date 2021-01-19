from sklearn.model_selection import train_test_split
from sklearn import ensemble

import matplotlib
import numpy as np


#----------------------------------main-------------------------------------#

data=np.genfromtxt("train.csv", delimiter=",", dtype=None, encoding=None)
x_train=data[1:,0:-1]    #  dados: da segunda à ultima linha, da primeira à penúltima coluna  
y_train=data[1:,-1]      # classe: da segunda à ultima linha, só última coluna

atributos = data[0,0:-1]

data=np.genfromtxt("test.csv", delimiter=",", dtype=None, encoding=None)
x_test=data[1:,0:-1]    #  dados: da segunda à ultima linha, da primeira à penúltima coluna  
y_test=data[1:,-1]   

#validation data 
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, random_state=1) #default test_size=25


print(f"Atributos: {atributos}\n\n")
print(f"X_TRAIN: {x_train},\n\n Y_TRAIN: {y_train},\n\n")
print(f"X_VALIDATION: {x_validation},\n\n Y_VALIDATION: {y_validation},\n\n")
print(f"X_TEST {x_test},\n\n Y_TEST: {y_test}\n\n")
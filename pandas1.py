import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

train_data = pd.read_csv('train.csv', header=0)

#remove ID and encode Programs
train_data.drop('Id',axis=1,inplace=True) 

print(train_data.head())

obj_train_data = train_data.select_dtypes(include=['object']).copy()
print(obj_train_data['Program'].value_counts())

enconde = {"Program": {"Informatics": 0, "Nursing": 1, "Management": 2, "Biology": 3}}

train_data = train_data.replace(enconde)
print(train_data.head())


#split treino em x_train e y_train
y_train = train_data[['Failure']].copy()
print(y_train)

train_data.drop('Failure',axis=1,inplace=True)
x_train = train_data.copy()
print(x_train)

#x_train = x_train.copy()
#x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, random_state=1) #default test_size=25
#y_train.ndim
#x_train.ndim

clf = GradientBoostingClassifier()
clf.fit(x_train, y_train.values.ravel())





# Read in the CSV file and convert "?" to NaN

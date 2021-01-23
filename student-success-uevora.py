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

def data_treatment(file):

    data = pd.read_csv(file, header=0)
    #print(file)
    #print(data)
    if(file=='train.csv'):

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
    
    else:
        
        save = data['Id'].copy()
        data.drop('Id',axis=1,inplace=True) 
        enconde = {"Program": {"Informatics": 0, "Nursing": 1, "Management": 2, "Biology": 3}}
        data = data.replace(enconde)

        return data,save

#----------------------------------main-------------------------------------#

#data_treatment
x_train,y_train = data_treatment('train.csv')
x_test,id_column = data_treatment('test.csv')

#clf = GradientBoostingClassifier(n_estimators=2000, max_depth=1)

k=920
min_estimators = 200
max_estimators = 2000
inc_estimators = int((max_estimators-min_estimators)/10)

#print(inc_estimators)
#for k in range(min_estimators,max_estimators+1,inc_estimators):

print(k)
clf = GradientBoostingClassifier(n_estimators=k,max_depth=1)
scores = cross_val_score(clf, x_train, y_train.values.ravel(), cv=10, scoring='accuracy')
print(scores.mean())



clf.fit(x_train, y_train.values.ravel())

result = clf.predict(x_test)
result = pd.DataFrame(data=result)
result.columns = ['Failure']
result.insert(loc=0,column ='Id',value=id_column)

result.to_csv(r'result.csv', index = False)



#clf.fit(x_train, y_train.values.ravel())

#final_score = clf.score(x_pred, y_pred) 
#print(f"final_score: {final_score*100}")

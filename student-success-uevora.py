import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

#prep
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
#models
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


#pred libraries
#from sklearn.cross_pred import StratifiedKFold
from IPython.display import display
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import warnings

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
clf = GradientBoostingClassifier(n_estimators=138, learning_rate=0.1, max_depth=8, min_samples_split=0.6, min_samples_leaf=7, max_features=4)
scores = cross_val_score(clf, x_train, y_train.values.ravel(), cv=10, scoring='accuracy')
print(scores.mean())




parameters = {
    "loss":["deviance"],
    "learning_rate": [0.1],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[3,5,8],
    "criterion": ["friedman_mse",  "mae"],
    "subsample":[0.5, 0.618, 1.0],
    "n_estimators":[138]
    }



parameters_K = {
    "n_neighbors":np.arange(1,201,50),
    "weights": ["uniform",  "distance"],
    "leaf_size": np.arange(1,501,50)
    }

parameters_L = {
    "C":np.arange(1,701,50),
    "penalty": ["l1","l2"],
    "solver": ["liblinear",],
    "tol": [1e-4],
    "max_iter": np.arange(1000,7001,1000),
    "intercept_scaling":np.arange(1,701,50),
    "multi_class":["auto"],
    }
    
lr = LogisticRegression(C = 1,
                          penalty='l1', 
                          solver='liblinear',  
                          tol=0.0001, 
                          max_iter=3000, 
                          intercept_scaling=1.0, 
                          multi_class='auto', 
                          random_state=42)

from sklearn.ensemble import BaggingClassifier

parameters_B = {
    "n_estimators":np.arange(1,500,50),
    "max_samples":np.arange(1,500,50),
    }

clf = GridSearchCV(BaggingClassifier(), parameters_B,scoring='accuracy',refit=True,cv=2, n_jobs=-1)
clf.fit(x_train, y_train.values.ravel())

scores = cross_val_score(clf, x_train, y_train.values.ravel(), cv=10, scoring='accuracy')
print(scores.mean())



result = clf.predict(x_test)
result = pd.DataFrame(data=result)
result.columns = ['Failure']
result.insert(loc=0,column ='Id',value=id_column)

result.to_csv(r'result.csv', index = False)



#clf.fit(x_train, y_train.values.ravel())

#final_score = clf.score(x_pred, y_pred) 
#print(f"final_score: {final_score*100}")

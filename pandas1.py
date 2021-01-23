import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

#prep
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
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

def best_parameters():
    return 0;
#----------------------------------main-------------------------------------#

#data_treatment
x_train,y_train = data_treatment('train.csv')
x_test,id_column = data_treatment('test.csv')

##############################################################################





min_estimators = 200
max_estimators = 2000
inc_estimators = int((max_estimators-min_estimators)/10)
n_estimators = range(min_estimators,max_estimators+1,inc_estimators)
vali_results = []

for estimator in n_estimators:
    model = GradientBoostingClassifier(n_estimators=estimator)

    scores = cross_val_score(model, x_train, y_train.values.ravel(), cv=5, scoring='accuracy')
    print(f"estimator: {estimator},accuracy: {scores.mean()}")
    vali_results.append(scores.mean())

   


from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(n_estimators, vali_results, 'b', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("AUC score")
plt.xlabel('n_estimators')
plt.show()

















"""
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

"""




















##############################################################################

clf.fit(x_train, y_train.values.ravel())

result = clf.predict(x_test)
result = pd.DataFrame(data=result)
result.columns = ['Failure']
result.insert(loc=0,column ='Id',value=id_column)

result.to_csv(r'result.csv', index = False)

"""
#split train to create pred data 
kf = KFold(n_splits=10)
#print(kf.get_n_splits(x_train))

for train_index, test_index in kf.split(x_train):
    print("TRAIN:", train_index, "TEST:", test_index)
    new_x_train, new_x_test = x_train[train_index], x_train[test_index]
    new_y_train, new_y_test = y_train[train_index], y_train[test_index]
#x_train, x_pred, y_train, y_pred = train_test_split(x_train, y_train, random_state=1) #default test_size=25
"""


#clf.fit(x_train, y_train.values.ravel())

#final_score = clf.score(x_pred, y_pred) 
#print(f"final_score: {final_score*100}")

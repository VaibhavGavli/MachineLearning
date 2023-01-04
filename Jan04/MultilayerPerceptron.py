import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


bank=pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Bankruptcy\Bankruptcy.csv",index_col=0)


X=bank.drop(['D','YR'],axis=1)
y=bank['D']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    train_size=0.7,
                                                    random_state=2022)

mm=MinMaxScaler()
mlp=MLPClassifier(activation='logistic',random_state=2022)
pipe=Pipeline([('MM',mm),('MLP',mlp)])
pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = pipe.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

####################### Grid Search CV ##################
pipe=Pipeline([('MM',mm),('MLP',mlp)])
print(pipe.get_params())

params={'MLP__hidden_layer_sizes':[(20,10,5),(10,5),(50,)],
        'MLP__activation':['tanh','logistic','identity'],
        'MLP__learning_rate':['constant','invscaling'],
        'MLP__learning_rate_init':[0.001,0.3,0.5]}

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
gcv=GridSearchCV(pipe,param_grid=params,verbose=3,
                scoring='roc_auc',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


#####Concrete Strength ####

from sklearn.neural_network import MLPRegressor
concrete =pd.read_csv(r"C:\Kaustubh Vaibhav\Advance Analystics\Cases\Concrete Strength\Concrete_Data.csv")
X =concrete.drop('Strength',axis=1)
y=concrete['Strength']

mm=MinMaxScaler()
mlp=MLPRegressor(random_state=2022)
pipe=Pipeline([('MM',mm),('MLP',mlp)])
params={'MLP__hidden_layer_sizes':[(6,4,3),(7,5),(10,)],
        'MLP__learning_rate':['constant','invscaling','adaptive'],
        'MLP__learning_rate_init':[0.001,0.3,0.5]}

kfold=KFold(n_splits=5,shuffle=True,random_state=2022)
gcv=GridSearchCV(pipe,param_grid=params,verbose=3,
                scoring='r2',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


####################### HR ####################
#########
hr = pd.read_csv(r"C:\Kaustubh Vaibhav\Advance Analystics\Cases\human-resources-analytics\HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first= True )
X = dum_hr.drop('left', axis = 1)
y = dum_hr['left']


X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)

mm=MinMaxScaler()
mlp=MLPClassifier(activation='logistic',random_state=2022)
pipe=Pipeline([('MM',mm),('MLP',mlp)])
pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = pipe.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

####################### Grid Search CV ##################
pipe=Pipeline([('MM',mm),('MLP',mlp)])
print(pipe.get_params())

params={'MLP__hidden_layer_sizes':[(20,10,5),(30,20,10),(40,30,10)],
        'MLP__activation':['tanh','logistic','identity'],
        'MLP__learning_rate':['constant','invscaling'],
        'MLP__learning_rate_init':[0.001,0.3,0.5]}

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
gcv=GridSearchCV(pipe,param_grid=params,verbose=3,
                scoring='roc_auc',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)
#coding: utf-8

'''
TPOT_GBR
'''
import csv
import math
import pandas as pd
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score,KFold
from sklearn import preprocessing, svm
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit


# input data
data_input=pd.read_csv('SI-ML_PtC_Database.csv',sep=',')

labels=data_input['Overpotential']#[:,np.newaxis]
features=data_input.drop('Overpotential', axis=1).drop('DOI', axis=1)

X_train,X_test,Y_train,Y_test=train_test_split(features, labels, test_size=0.2,random_state=99) 
reg = GradientBoostingRegressor(alpha=0.75, learning_rate=0.1, loss="ls", max_depth=8, max_features=0.6000000000000001, min_samples_leaf=1, min_samples_split=2, n_estimators=100, subsample=0.7000000000000001)
reg.fit(X_train, Y_train)

Y_pred =  reg.predict(X_test)
Y_pred_2 =  reg.predict(X_train)

result1=pd.DataFrame(columns=['y_test','y_pred'])
result2=pd.DataFrame(columns=['y_train','y_train_pred'])
result1['y_test']=Y_test
result1['y_pred']=Y_pred
result2['y_train']=Y_train
result2['y_train_pred']=Y_pred_2
print("Train Accuracy : %.4g" % sk.metrics.r2_score(Y_train, Y_pred_2))
print("Test Accuracy : %.4g" % sk.metrics.r2_score(Y_test, Y_pred))

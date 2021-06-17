#coding: utf-8

'''
TPOT
'''

import csv
import math
import pandas as pd
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import xgboost
from tpot import TPOTClassifier
from tpot import TPOTRegressor
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import svm

# input data
data_input=pd.read_csv('SI-ML_PtC_Database.CSV',sep=',')

labels=data_input['Overpotential']#[:,np.newaxis]
features=data_input.drop('Overpotential', axis=1).drop('DOI', axis=1)

X_train,X_test,y_train,y_test=train_test_split(features, labels, test_size=0.2, random_state=0)

tpot = TPOTRegressor(random_state=0,verbosity=2)

tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_pipeline.py')

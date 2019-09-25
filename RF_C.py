# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:43:51 2019

@author: Shriyash Shende
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
t = pd.read_csv('Fraud_check.csv')
t.info()
t.describe()
t.columns
t1 = pd.get_dummies(t, drop_first = True)
t1.rename(columns={'Taxable.Income':'TaxableIncome'}, inplace=True)
t1["Fraud_Check"] = ""
for i, TaxableIncome in enumerate(t1.TaxableIncome):
    if TaxableIncome <= 30000:
        t1.Fraud_Check[i]= "RISKY"
    else:
        t1.Fraud_Check[i]= "GOOD"
          
t1 = t1.drop(['TaxableIncome'], axis = 1)          
X = t1.drop(["Fraud_Check"], axis = 1)
Y = t1["Fraud_Check"]


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)


### Random forest
from sklearn.ensemble import RandomForestClassifier
ml = RandomForestClassifier(n_jobs=2,oob_score=True,n_estimators=1000,criterion="entropy")
ml.fit(X_train,Y_train)
ml.oob_score_ 
Y_pred = ml.predict(X_test)
#Checking Accuracy
from sklearn.metrics import confusion_matrix
cn = confusion_matrix(Y_test,Y_pred)
cn
per = cn[0,0] + cn[1,1]
p = per + cn[0,1] + cn[1,0]
per / p


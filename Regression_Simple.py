# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#import array as ar
import math as mat



dtst = pd.read_csv("reg_simple.csv")

X1 = dtst.iloc[:,:-1]
Y1 = dtst.iloc[:,-1]

plt.scatter(X1, Y1)
plt.xlabel("heure_rev_independance_var")
plt.ylabel("note : dependance variable")
plt.style.use(['dark_backgroud', 'fast'])
plt.show()


X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, test_size=0.2, random_state=0)

Regresseur = LinearRegression()

Regresseur.fit(X_train, Y_train)

#etablir une prediction
Y_Prediction = Regresseur.predict(X_test)

plt.plot(X_test, Y_Prediction, c='r')
plt.plot(X1, Y1)
plt.show()

plt.plot(X_test, Y_Prediction, c='g')
plt.scatter(X_test, Y_test, c='r')
#plt.scatter(X1, Y1, c='b')
plt.show()

type(X_test)


Regresseur.predict([[18]]) # --> array([60.51706037])
"""
    La prediction sur la valeur 18 a donné une erreur.
    parce dans notre dataset (on a la chance d'avoir un dataset petit'),
    on a un écart entre la valeur correspondante '66' et celle prédite.
    Pour remedier à ce problème, on implémente la fonction suivante :
        couramment connue sous le nom d'erreur quadratique moyenne
"""
def rmse(val_obs, val_pred):
    y_actual = np.array(val_obs)
    y_pred = np.array(val_pred)
    error = (y_actual - y_pred)**2
    error_mean = round(np.mean(error))
    error_sq = mat.sqrt(error_mean)
    
    return error_sq
    
# Version condensée :
def rmse_cds(val_obs, val_pred):
    return mat.sqrt(round(np.mean((np.array(val_obs)-np.array(val_pred))**2)))

rmse(66, Regresseur.predict([[18]]))
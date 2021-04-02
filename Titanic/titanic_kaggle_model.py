#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 23:44:11 2019

@author: felipe
"""
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"Estabelecendo conexão"
engine = create_engine("mysql+pymysql://root:123456@127.0.0.1/titanic", echo=True)
conn = engine.connect()

"Importando o  teste"
from sqlalchemy import text
t = text("SELECT * FROM titanic.passageiros_test")
result = conn.execute(t)
rows = result.fetchall()

"Formas de transferencia para vetores utilizando numpy"
passengers_test = np.reshape(rows, (len(rows),len(rows[0])))
#passengers_test = np.array(rows)

"Query para train"
t_2 = text("SELECT * FROM titanic.passageiros")
result_2 = conn.execute(t_2)
rows_2 = result_2.fetchall()
passengers = np.array(rows_2)

"Pemutação da variável dependente para o fim da tabela"
permutacao = [0, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
i = np.argsort(permutacao)
passengers = passengers[:, i]

"Construindo o modelo"
X_test = passengers_test[:, 1:]
X_train = passengers[:, 1:]
y_train = X_train[:, [10]]

df_train = pd.DataFrame(X_train)
df_test = pd.DataFrame(X_test)
df_y = pd.DataFrame(y_train)

df_train.isnull().sum()*100/df_train.shape[0]

"Tratando dados do treino"
X_train = np.delete(X_train, [1, 8, 10], 1)  #Removendo nomes #Removendo variavel dependente #Removendo a informação da cabine

"1 = sex male(1); 5 = Ticket; 7 = Embarked"
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Xtrain1 = LabelEncoder()
labelencoder_Xtrain5 = LabelEncoder()
labelencoder_Xtrain7 = LabelEncoder()
X_train[:, 1] = labelencoder_Xtrain1.fit_transform(X_train[:, 1]) 
X_train[:, 5] = labelencoder_Xtrain5.fit_transform(X_train[:, 5])
X_train[:, 7] = labelencoder_Xtrain7.fit_transform(X_train[:, 7])

"""
ohe_Xtrain = OneHotEncoder(categorical_features = [7] , sparse = True)
X_train = ohe_Xtrain.fit_transform(X_train).toarray()
"Dummy variable"
X_train = X_train[:, 1:]"""

from sklearn.preprocessing import StandardScaler
sc_train = StandardScaler()
X_train = sc_train.fit_transform(X_train)

"Tratando dados test"
X_test = np.delete(X_test, [1, 8], 1)
labelencoder_testX1 = LabelEncoder()
labelencoder_testX5 = LabelEncoder()
labelencoder_testX7 = LabelEncoder()
X_test[:, 1] = labelencoder_testX1.fit_transform(X_test[:, 1])
X_test[:, 5] = labelencoder_testX5.fit_transform(X_test[:, 5])
X_test[:, 7] = labelencoder_testX7.fit_transform(X_test[:, 7])

"""
ohe_test = OneHotEncoder(categorical_features = [7], sparse=True)
X_test = ohe_test.fit_transform(X_test).toarray()
X_test = X_test[:, 1:] This is wrong, but"""

sc_test = StandardScaler()
X_test = sc_test.fit_transform(X_test)


"Modelo com Decision Tree"
from sklearn.tree import DecisionTreeClassifier
reg_tree = DecisionTreeClassifier()
reg_tree.fit(X_train, y_train)

y_pred = reg_tree.predict(X_test)
df_pred = pd.DataFrame(y_pred)


"Contando valores no array"  
unique, counts = np.unique(y_pred, return_counts = True)
dict(zip(unique, counts))

"Colocando em um csv"
aux = np.arange(892, 1310, 1).astype(int) 
y_test = np.array(list(zip(aux, y_pred)))

import csv
with open("Results_titanic.csv", "w", newline = "") as f:
    escrever = csv.writer(f)
    escrever.writerows(y_test)
   


    



"""
"Construindo um modelo a partir de uma regressão linear múltipla"
from sklearn.linear_model import LinearRegression
reg_lin = LinearRegression()
reg_lin.fit(X_train, y_train)


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(X_train)
reg_lin.fit(X_poly, y_train)

y_pred = reg_lin.predict(poly.fit_transform(X_test))
df_pred = pd.DataFrame(y_pred)

y_p_inverse = sc_test.inverse_transform(y_pred)

y_k = reg_lin.predict(poly.fit_transform([[6]]))

from sklearn.pipeline import Pipeline
pipeline = Pipeline([
                    ('poly', PolynomialFeatures(degree = 2)),
                    ('linreg', LinearRegression())
                    ])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
"""






 






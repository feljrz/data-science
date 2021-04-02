#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 23:56:08 2019

@author: felipe
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats


#Algum bug desconhecido indica que o dataframe não é o mesmo, porém em type(df) apresenta a contradição. Não sendo possível acessar métodos, não apresentado documentação
#df = pd.read_csv("/home/felipe/Documents/IA/Databases/Autos/backup_nomeado.csv")
df = pd.read_csv("database/imports-85.data", header = None)
headers = ['symboling','normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
         'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke',
         'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
#Estabelecendo cabeçalho
df.columns = headers
df.columns
"""
X = df2.iloc[:,:].values
df = pd.DataFrame(X) #problema pois todos irão ter tipo object
"""


#Verificando infos
df.info()
df.dtypes #verificar os tipos de dados de cada coluna
df.describe(include = 'all') #visualizando principais dados da tabela


#Salvando o dataset para preservação
#df.to_csv("/home/felipe/Documents/IA/Databases/Autos/backup_nomeado.csv")

#retirando valores nulos, inplace = True permite a alteração no dataframe
df['price'].dropna(axis = 0, inplace = True) #Descobrir pq nao ta dropando
df['price'].replace(to_replace = '?', value = 'NaN', inplace = True)
df = df.dropna(axis = 0, subset = ['price'])
linhas_nulas_preco = df.loc[df['price'] == 'NaN', 'price'].index.to_list()
df.drop(linhas_nulas_preco, axis = 0, inplace = True)




#Com Imputer
from sklearn.preprocessing import Imputer
imp = Imputer()



"""
Querys no dataframe
indices = df.price[df['price'] == '?'].index.to_list()
indices2 = df.loc[df['price']=='?', 'price'].index.to_list()
menores_que = df.loc[(df['length']< 150) & (df['height'] < 54),['make','price']]
df['price'][(df['height'] < 54) & (df['length'] < 150)]
query = df.query('height < 54 and length < 150')"""


#Binning a data 
df['price'] = df['price'].astype('int')
bins = np.linspace(df['price'].min(), df['price'].max(), 4)
groups = ['Low', 'Medium', 'High']
df['price-binned'] = pd.cut(df['price'], bins = bins, labels = groups, include_lowest = True)

#Variaveis categoricas OneHotEncode
dummies = pd.get_dummies(df['fuel-type'])
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = LabelEncoder()
df['fuel-type'] = encoder.fit_transform(df['fuel-type'])
#ohe = OneHotEncoder(categorical_features = df['fuel-type'])
#Trabalho do OneHotEncoder
dummies.columns = ['gas', 'diesel']
df = pd.concat([dummies, df], axis = 1)
df = df.drop(labels = 'fuel-type',axis = 1)

dw_counts = df["drive-wheels"].value_counts()
dw_counts.rename(columns = {'drive-wheels', 'value_counts'}, inplace = True) #Descobrir pq não esta funcionando

#Alguns gráficos
sns.boxplot(x = df['drive-wheels'], y = df['price'], data = df)
sns.scatterplot(x= df['engine-size'], y = df['price'])
plt.xlabel('Tamanho do motor')
plt.ylabel('Preço')
engine-size'
engine_size = np.array(df['engine-size'].values)
engine_size = engine_size.reshape(df['engine-size'].size, 1)

#Groupby e pivot
df_aux = df[['price', 'drive-wheels', 'body-style']]
df_group = df_aux.groupby(['drive-wheels', 'body-style'], as_index = False).mean()
df_pivot = df_group.pivot(index = 'drive-wheels', columns = 'body-style', values = 'price')
plt.bar(x  = df_group['body-style'], height = df_group['price'])

plt.pcolor(df_pivot, cmap='RdBu') #descobrir como colocar os indices
plt.colorbar()



#ANOVA fazendo uma análise do f-test e do pvalue
#Organizando grupo a ser analisado, analisaremos correlacao entre duas variaveis
df_make_price = df[['make', 'price']]
df_group_make = df_make_price.groupby(['make'], as_index = False).mean()
plt.bar(x = df_group_make['make'], height = df_group_make['price'].sort_values())
plt.xlabel('Marcas'); plt.ylabel('Preços'); plt.title('Preço dos carros por marca')
dict_group = dict(zip(list(df_group_make['make']), list(df_group_make['price'])))
#Existe o df_group_make.get_group('honda')['price'] porem nao está funcionando (Descobrir o porque)
ho = df_make_price.price[df_make_price['make'] == 'honda']
dg = df_make_price.price[df_make_price['make'] == 'dodge']
anova_results = stats.f_oneway(ho, dg)
#av = stats.f_oneway(df.loc[df['make'] == 'honda', ['price']], df.loc[df['make'] == 'jaguar', ['price']])
ho = df_make_price.price[df_make_price['make'] == 'honda']
jg = df_make_price.price[df_make_price['make'] == 'jaguar']
anova_results = stats.f_oneway(ho, jg)
#Calculando Correlação
#Convertendo milhas por galão em kilometros por litro
df['highway-mpg'] = df['highway-mpg'] * 0.425
df.rename(columns = {'highway-mpg':'km/l'}, inplace = True)
sns.regplot(x = df['km/l'], y= df['price'], data=df)
plt.ylim(0,)
sns.regplot(x = df['engine-size'], y= df['price'], data=df)
plt.ylim(0,)
#coeficiente de pearson coeficiente proximo de 1 indica uma relação linar e p_value baixo indica alta confiabilidade
coef, p_val = stats.pearsonr(df['engine-size'], df['price'])



#Separando em train e test
X = df[['engine-size', 'horsepower', 'km/l', 'peak-rpm']]
y = df[['price']]

#Resolvelndo missing values
from sklearn.preprocessing import Imputer
X = X.replace(to_replace = '?', value = 'NaN')
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X.iloc[:,:])
X = imputer.transform(X)
X = pd.DataFrame(X)

#Construindo um modelo Linear
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X, y)
a = lm.coef_
b = lm.intercept_

#Construindo um modelo polinomial
from sklearn.preprocessing import PolynomialFeatures
polym = PolynomialFeatures()
X_poly = polym.fit(X)


#análise dos resíduos e distribuição
sns.residplot(df['engine-size'], df['price'])
sns.distplot(df['price'], hist = False)








































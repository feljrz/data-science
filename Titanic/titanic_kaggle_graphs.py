#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:17:21 2019

@author: felipe
"""

from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

df = pd.read_csv("train.csv")

df = df.drop(['Ticket', 'Cabin'], axis = 1)

df = df.dropna()    #Drop NaN values

fig = plt.figure(figsize=(18, 6), dpi = 1600)
alpha = alpha_scatterplot = 0.2
alpha_bar_chart = 0.9

#plotando diferentes gráficos

ax1 = plt.subplot2grid((2,3), (0,0))

#plotando um gráfico de barras com informações de quem sobreviveu e quem não
df.Survived.value_counts().plot(kind = 'bar', alpha = alpha_bar_chart)
#ax1.set_xlim(-1, 2) resolvendo possível bug
plt.title("Distribuição de sobreviventes")

plt.subplot2grid(shape=(2,3), loc=(0,1))
plt.scatter(x=df.Survived, y=df.Age, alpha = alpha_scatterplot)
plt.ylabel("Age")
plt.grid(b=True, which='major', axis = 'y')
plt.title("Sobreviventes por Idade")

count = df.Pclass.unique()

df.Pclass.v

ax3 = plt.subplot2grid(shape=(2,3), loc=(0,2))
#plt.bar(df.Pclass.unique(), height = df.Pclass.value_counts())
df.Pclass.value_counts().plot(kind='barh', alpha = alpha_bar_chart)
#ax3.set_ylim(-1, len(df.Pclass.value_counts()))
plt.xlabel("aa")
plt.title("Distribuição de classes")


# MUITO IMPORTANTE, kde = kernel density estimation, simula a curva de distribuição de probabilidade  
plt.subplot2grid((2,3), (1,0), colspan=2)
df.Age[df.Pclass == 1].plot(kind='kde')
df.Age[df.Pclass == 2].plot(kind='kde')
df.Age[df.Pclass == 3].plot(kind='kde')
plt.xlabel("Idade")
plt.title("Distribuição de densidade de Idade X Classe")
plt.legend(('1 Classe', '2 Classe', '3 Classe'), loc='best')

ax5 = plt.subplot2grid((2,3), (1,2))
plt.bar(df.Embarked.unique(),df.Embarked.value_counts(), alpha=alpha_bar_chart)

fig = plt.figure(figsize=(18, 6))
#ANALISANDO SOBREVIVENTES

df_male = df.Survived[df.Sex == 'male'].value_counts().sort_index()
df_female = df.Survived[df.Sex == 'female'].value_counts().sort_index()

ax1 = fig.add_subplot(121)
df_male.plot(kind = 'barh', label = 'Masculino', alpha = 0.55)
df_female.plot(kind = 'barh', color = '#FF00FF', label = 'Feminino', alpha = 0.55)
plt.title("Sobrevivendes em relação à Genero")
plt.legend(loc = 'best')

#Quem sobreviveu proporcionalmente
ax2 = fig.add_subplot(122)
(df_male/float(df_male.sum())).plot(kind = 'barh', alpha = 0.55, color = '#0000FF')
(df_female/float(df_male.sum())).plot(kind = 'barh', alpha = 0.55, color = '#FF4500')
plt.title("Sobreviventes proporcionalmente com relação a gênero")
plt.legend(("Masculino", "Feminino"),loc = 'best')

"""
#Sobreviventes em ralação à classe
fig_2 = plt.figure(figsize = (18, 6))

dict_class = {
'Primeira': np.array(df.Survived[df.Pclass == 1].value_counts().sort_index()),
'Segunda': np.array(df.Survived[df.Pclass == 2].value_counts().sort_index()),
'Terceira': np.array(df.Survived[df.Pclass == 3].value_counts().sort_index())}

ax1 = fig_2.add_subplot(121)

plt.bar(x = dict_class.keys(), height =  width = 0.5)
"""

fig = plt.figure((18, 4), dpi = 1600)
alpha_level = 0.6

ax1 = fig.add_subplot(141)
female_highclass = df.Survived[df.Sex == 'female'][df.Pclass != 3].value_counts()
female_highclass.plot(kind='bar', label = 'female, highclass', color = '#FA2479', alpha = alpha_level)
ax1.set_xticklabels(['Survived', 'Died'],rotation = 0 )
plt.legend(loc='best')

ax2 = fig.add_subplot(142, sharey = ax1)
female_lowclass = df.Survived[df.Sex == 'female'][df.Pclass == 3].value_counts()
female_lowclass.plot(kind = 'bar', label = 'female, low class', color = 'pink', alpha = alpha_level)
plt.legend(loc = 'best')

ax3 = fig.add_subplot(143, sharey = ax1)
male_highclass = df.Survived[df.Sex == 'male'][df.Pclass != 3].value_counts()
male_highclass.plot(kind = 'bar', label = 'male, highclass', color = 'lightblue',alpha = alpha_level)
ax3.set_xticklabels(['Morreram', 'Sobreviveram'], rotation = 0)
plt.legend(loc = 'best')

ax4 = fig.add_subplot(144, sharey = ax1)
male_lowclass = df.Survived[df.Sex == 'male'][df.Pclass == 3].value_counts()
male_lowclass.plot(kind = 'bar', label = 'male, lowclass', color = 'blue', alpha = alpha_level)
ax4.set_xticklabels(['Morreram', 'Sobreviveram'], rotation = 0)
plt.legend(loc = 'best')




"""
Mudar tick nas labels
fig, ax = plt.subplots()

# We need to draw the canvas, otherwise the labels won't be positioned and 
# won't have values yet.
fig.canvas.draw()

labels = [item.get_text() for item in ax.get_xticklabels()]
labels[1] = 'Testing'

ax.set_xticklabels(labels)

plt.show()


fig2, ax5 = plt.subplots()
male_lowclass.plot(kind = 'bar', label = 'male, lowclass', color = 'blue', alpha = alpha_level)
labels = [item.get_text() for item in ax5.get_xticklabels()]
labels[0] = 'Morreram'
labels[1] = 'Sobreviveram'
ax5.set_xticklabels(labels, rotation = 0)
"""


#!/usr/bin/env python
# coding: utf-8


import pandas as pd 
import catboost as cb 
import numpy as np
from matplotlib import pyplot as plt


# # Busqueda de la cantidad de iteraciones



train_values = pd.read_csv('train_values_catboost.csv', index_col='building_id')
train_labels = pd.read_csv('train_labels.csv', index_col='building_id')


categoricas = []
for col in train_values.columns:
    if ((col != 'min_geo_id') & (col != 'max_geo_id')):
        categoricas.append(col)
        categoricas.append(train_values.columns.get_loc(col))
        train_values[col] = train_values[col].astype("category")



pool = cb.Pool(data=train_values,
                label=train_labels,
                cat_features=categoricas)

# Para correr en gpu habria que agregar a params
#  "task_type":"GPU"
params = {"loss_function":'MultiClass', "eval_metric":'TotalF1:average=Micro', "iterations":4000,
        "random_seed":2021}
resultados = cb.cv(pool, params, fold_count=12)


print("Cantidad de iteraciones que mejor funciono:  ", end='')
print(resultados["test-TotalF1:average=Micro-mean"].idxmax())


plt.plot(resultados['iterations'],resultados["test-TotalF1:average=Micro-mean"])
plt.xlabel("Cantidad de iteraciones",  fontsize=15)
plt.ylabel("F1 Score promedio en el set de test",  fontsize=15)
plt.title('F1 Score promedio en el set de test segun la cantidad de iteraciones\n'
         "en Catboost con CrossValidation y 10 folds",  fontsize=20)
plt.show()
plt.savefig("Cantidad de iteraciones.jpg")


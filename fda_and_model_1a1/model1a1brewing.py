"""
Classification 
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, auc, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt



def run_model_info(tab,BEERSDIC,train,test,n_clusters_list):
 tab = tab.loc[test.Cliente.values,:].copy()
 i = 1
 for marca, beer in zip(test.columns[1:],BEERSDIC.keys()):
  name_columns_features_beer = [x for x in tab.columns if ((beer in x ) and ('cluster' not in x)) ]
  name_columns_features_all  = [x for x in tab.columns if (('_all' in x ) and ('cluster' not in x))]
  if ((i==1) or (i==4) or (i==5)):
    name_columns_categ = [x for x in tab.columns if (f'cluster_{beer}'  in x)] + ['Gerencia2','SubCanal2', 'Categoria', 'Nevera']
    name_columns_features_train = name_columns_features_beer[:-1]  + name_columns_categ + name_columns_features_all[:-1]
    name_columns_features_test = name_columns_features_beer[1:] + name_columns_categ + name_columns_features_all[1:] 
    param_grid = [{"max_leaf_nodes":[2,3,4,5]}]
    X_train = tab[name_columns_features_train].copy()
    X_train = pd.get_dummies(data=X_train, columns=name_columns_categ)
    X_test = tab[name_columns_features_test].copy()
    X_test = pd.get_dummies(data=X_test, columns=name_columns_categ)
    print(X_train.columns)
    y = tab[f'2020-09_{beer}'].values
  else:
    name_columns_categ = [x for x in tab.columns if (f'cluster_{beer}'  in x)] + ['Gerencia2','SubCanal2', 'Categoria', 'Nevera']
    name_columns_features_train = name_columns_features_beer[:-1] + name_columns_features_all[:-1] + name_columns_categ
    name_columns_features_test = name_columns_features_beer[1:] + name_columns_features_all[1:] + name_columns_categ
    X_train = tab[name_columns_features_train].copy()
    X_train = pd.get_dummies(data=X_train, columns=name_columns_categ)
    X_test = tab[name_columns_features_test].copy()
    X_test = pd.get_dummies(data=X_test, columns=name_columns_categ)
    print(X_train.columns)
    y = tab[f'2020-09_{beer}'].values
  model = RandomForestClassifier()
  param_grid = [{"max_depth":[20,25,30],
               "max_features":[20,25],
               "max_leaf_nodes":[ 20, 30, 25],
               "n_estimators":[25, 20, 30 ]}]
  grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, return_train_score=True, scoring='roc_auc')
  grid_search.fit(X_train, y)
  plt.plot(grid_search.best_estimator_.feature_importances_)
  plt.show()
  print('***********************************************************************')
  y_pred = grid_search.best_estimator_.predict_proba(X_train)[:,1]
  test[f'{marca}'] = grid_search.best_estimator_.predict_proba(X_test)[:,1]
  i +=1
return test
 

"""
Functions data functional analysys for Brewing Data Cup
"""
import numpy as np
import pandas as pd  
import seaborn as sns
import matplotlib.pyplot as plt
import skfda


def cluster_fda(table,num_clusters,marca):
  data_fda=skfda.FDataGrid(table)
  kmeans = skfda.ml.clustering.KMeans(n_clusters=num_clusters)
  kmeans.fit(data_fda)
  clusters_df=pd.DataFrame(kmeans.labels_,index=table.index)
  table_cluster=table.merge(clusters_df,how='inner',right_index=True, left_index=True)
  table_cluster.rename(columns={0:f'cluster_{marca}'},inplace=True)
  return table_cluster

def get_table(BEERSDIC, train, n_clusters_list):
  list_tables = []
  for index, beer in enumerate(BEERSDIC.keys()):
    tabla = table_from_df(train, beer_list = BEERSDIC[beer])
    tabla.columns = [ f'{str(x)[:7]}_{beer}' for x in tabla.columns.ravel()]
    n_clu = n_clusters_list[index]
    tab_cluster = cluster_fda(tabla,n_clu,beer)
    list_tables.append(tab_cluster)
  tabla = table_from_df(train,all_volumens=True)
  tabla.columns = [ f'{str(x)[:7]}_all' for x in tabla.columns.ravel()]
  n_clu = n_clusters_list[-1]
  tab_cluster = cluster_fda(tabla,n_clu,'all')
  list_tables.append(tab_cluster)
  tab = pd.concat(list_tables, axis=1)
  return tab
 
def table_from_df(data,beer_list = ["Marca_20","Cupo_3","CapacidadEnvase_9"], var='Volumen',all_volumens =False):
  if not all_volumens:
    mask_marca = (data["Marca2"]==beer_list[0])
    mask_cupo =  (data["Cupo2"]==beer_list[1])
    mask_capac = (data["CapacidadEnvase2"]==beer_list[2])
    mask = (mask_marca & mask_cupo & mask_capac)
    data_filter = data[mask]
  else:
    data_filter = data
  clientes = pd.DataFrame(data=list(data["Cliente"].unique()),columns=['Cliente'])
  table = pd.pivot_table(data=data_filter, values=var,
                         index ='Cliente', columns='date')
  table = clientes.merge(table,how = 'left', left_on='Cliente', right_index=True)
  table.fillna(0, inplace=True)
  return table
  
def plot_mean_clusters(tabclu_o):
  tabclu=tabclu_o.copy()
  columns = list(tabclu.columns)
  columns.pop()
  columns.append('cluster')
  tabclu.columns =columns
  clusters=sorted(list(tabclu.cluster.value_counts().index.array))
  sns.set_style("darkgrid")
  fig, ax=plt.subplots(figsize=(9,7))
  for cluster in clusters:
    tab_c=tabclu[tabclu.cluster==cluster].copy()
    tab_c.drop(columns=['cluster','Cliente'],inplace=True)
    fda_tab_c=skfda.FDataGrid(tab_c)
    mean_periodos=skfda.exploratory.stats.mean(fda_tab_c)
    lab='Cluster '+str(cluster)
    ax.plot(mean_periodos.data_matrix[0,:,0],'-o',label=lab)
  tabclu.drop(columns=['cluster','Cliente'],inplace=True)
  fda_all=skfda.FDataGrid(tabclu)
  mean_all=skfda.exploratory.stats.mean(fda_all)
  ax.plot(mean_all.data_matrix[0,:,0],'--o',label='All data')
  dates=[str(x) for x in tabclu.columns]
  ax.set_xticks(np.arange(len(dates)))
  ax.set_xticklabels(dates,rotation=90)
  plt.title('Comportamiento Volumenes')
  plt.legend()
  plt.show()
  
def plot_inertias(tabla):
  list_inertias = []
  num_clusters_list =np.arange(2,15)
  for num_clusters in num_clusters_list:
    data_fda=skfda.FDataGrid(tabla.copy())
    kmeans = skfda.ml.clustering.KMeans(n_clusters=num_clusters)
    kmeans.fit(data_fda)
    list_inertias.append(kmeans.inertia_)
  plt.plot(num_clusters_list,list_inertias,'*-')





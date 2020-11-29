from pathlib import Path
import pandas as pd
import os

class DataSet():

  BASEPATH = Path("/content/drive/MyDrive/brewing")
  DATAPATH = BASEPATH / "data"
  FILENAMES = {"clients":"Input1_clientes_estructura.csv",
             "train_sells":"Input2_clientes_venta.csv",
             "test_sells":"Input3_clientes_test.csv"}
  BEERSDIC ={"beer_1":["Marca_20","Cupo_3","CapacidadEnvase_9"],
             "beer_2":["Marca_16","Cupo_2","CapacidadEnvase_10"],
             "beer_3":["Marca_9","Cupo_3","CapacidadEnvase_12"],
             "beer_4":["Marca_38","Cupo_2","CapacidadEnvase_10"],
             "beer_5":["Marca_39","Cupo_2","CapacidadEnvase_10"]}
  def __init__(self):
    self.clients_df = pd.read_csv(self.DATAPATH/self.FILENAMES["clients"], sep=";")
    self.train_df = pd.read_csv(self.DATAPATH/self.FILENAMES["train_sells"], sep=";")
    self.train_df["date"] = (self.train_df["Año"].astype('str') + '/' 
        + self.train_df["Mes"].astype('str')).astype("datetime64")
    self.train_df = self.train_df.astype({"Cliente":'category'})
    self.test_df = pd.read_csv(self.DATAPATH/self.FILENAMES["test_sells"], sep=";")
  
  def get_full_features(self, train=True):
    filename = 'train_features_full.csv' if train else 'test_features_full.csv'
    if not os.path.exists(self.DATAPATH/filename):
      features = self.get_features(train=train)
      extra_features = self._get_extra_features()
      full_features = features.merge(extra_features, how='left', on='Cliente')
      full_features.to_csv(self.DATAPATH/filename, index=False)
    else:
      full_features = pd.read_csv(self.DATAPATH/filename)
    return full_features

  def _get_extra_features(self):
    popular_marcas = ["Marca_1", "Marca_2", "Marca_4", "Marca_5", "Marca_6"]
    popular_masks = [self.train_df["Marca2"]==marca for marca in popular_marcas ]
    pivot_dfs = []
    for i, popular_mask in enumerate(popular_masks):
      pivot_df = pd.pivot_table(self.train_df.loc[popular_mask], values="Volumen", index="Cliente", columns="date", aggfunc='sum', fill_value=0)
      new_col_names = [f'marca_popular_{i}_{col.year}_{col.month}_vol' for col in pivot_df.columns]
      pivot_df.columns = new_col_names
      pivot_dfs.append(pivot_df)
    extra_features = pd.concat(pivot_dfs,axis=1)
    extra_features = extra_features.merge(pd.DataFrame(self.train_df["Cliente"].unique(), columns=["Cliente"]), how="right", on="Cliente")
    extra_features = extra_features.set_index("Cliente").fillna(0)
    return extra_features

  def get_features(self, train=True):
    filename = 'train_features.csv' if train else 'test_features.csv'
    if not os.path.exists(self.DATAPATH/filename):
      data = self._get_data()
      drop_test = ['marca_2_2019_5_vol',
                    'marca_3_2019_5_vol',
                    'marca_1_2020_7_vol',
                    'marca_4_2020_7_vol',
                    'marca_5_2020_7_vol'] # Note the columns on train have a lag!
      train_cols = list(data.columns[:-5])
      test_cols = [col for col in data.columns if col not in drop_test]
      cols = train_cols if train else test_cols
      features = data[cols]
      features.columns = train_cols # Note the columns on train have a lag!
      if not train:
        clientes_df = self.clients_df.drop(columns=["Regional2"]).merge(features, how='left', on="Cliente")
        features = self.test_df[["Cliente"]].merge(features, how='left', on='Cliente')
        features.fillna(0, inplace=True)
      features.to_csv(self.DATAPATH/filename, index=False)
    else:
      features = pd.read_csv(self.DATAPATH/filename)
    return features
    
  def _get_data(self):
    list_pivots_tables = []
    for i, beer in enumerate(self.BEERSDIC.keys()):
      mask_marca = (self.train_df["Marca2"]==self.BEERSDIC[beer][0])
      mask_cupo =  (self.train_df["Cupo2"]==self.BEERSDIC[beer][1])
      mask_capac = (self.train_df["CapacidadEnvase2"]==self.BEERSDIC[beer][2])
      mask = (mask_marca & mask_cupo & mask_capac)
      df = self.train_df.loc[mask].copy()
      pivot_tab = pd.pivot_table(df, values='Volumen', index='Cliente', columns='date')
      change_names = lambda n_col: f'marca_{i+1}_{n_col.year}_{n_col.month}_vol'
      pivot_tab.rename(columns=change_names, inplace=True)
      list_pivots_tables.append(pivot_tab)
    df_pivots = pd.concat(list_pivots_tables, axis=1)
    df_pivots.reset_index(inplace=True)
    data = self.clients_df.merge(df_pivots, left_on='Cliente', right_on = 'Cliente')
    data.fillna(0, inplace = True)
    data = data[self._sort_columns(data.columns)]
    return data
  
  def _sort_columns(self, df_columns):
    list_cols = ['Cliente','Gerencia2','SubCanal2','Categoria','Nevera']
    date_cols = [f"marca_{marca}_{year}_{month}_vol" for year in [2019, 2020] for month in range(1,13) for marca in range(1,6)]
    date_cols = [date_col for date_col in date_cols if date_col in df_columns]
    return list_cols + date_cols

  def get_labels(self):
    filename = 'labels.csv'
    if not os.path.exists(self.DATAPATH/filename):
      data = self._get_data()
      keep_cols = data.columns[-5:]
      labels = (data[keep_cols] > 0) * 1.0
      labels.to_csv(self.DATAPATH/filename, index=False)
    else:
      labels = pd.read_csv(self.DATAPATH/filename)
    return labels
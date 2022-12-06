import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt


#%%
#importing datasets

generation_data = pd.read_csv('/home/pauloguedes/series-temporais/datasets/kaggle/Plant_1_Generation_Data.csv')
weather_data = pd.read_csv('/home/pauloguedes/series-temporais/datasets/kaggle/Plant_1_Weather_Sensor_Data.csv')
generation_data.info()
#%%
#merging data

generation_data['DATE_TIME'] = pd.to_datetime(generation_data["DATE_TIME"])
weather_data['DATE_TIME'] = pd.to_datetime(weather_data["DATE_TIME"])

df = pd.merge(generation_data.drop(columns=['PLANT_ID']), weather_data.drop(columns=['PLANT_ID', 'SOURCE_KEY']), on='DATE_TIME')
#%%

df.isnull().sum()

#%%
pd.plotting.scatter_matrix(df, figsize=(15,15))

plt.show()

#%%

corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')

#%%

#Convert 'SOURCE_KEY' to numerical type
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df['SOURCE_KEY_NUMBER'] = encoder.fit_transform(df['SOURCE_KEY'])

#%%


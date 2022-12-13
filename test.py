import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from numpy import array
from pandas import read_csv
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from math import sqrt
from sklearn.metrics import mean_squared_error
#%%
#Function to split one variable time series sequences
def split_sequence(sequence_x, sequence_y, n_steps):
	X, y = list(), list()
	for i in range(len(sequence_x)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence_x)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence_x[i:end_ix], sequence_y[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

#Function to create series
def creates_series(series_x, series_y, n_steps, n_features,split_size):
    X = series_x.values
    Y = series_y.values
    train_x, test_x = X[0:split_size], X[split_size:]
    train_y, test_y = Y[:split_size], Y[split_size:]
    X_train, y_train = split_sequence(train_x,train_y, n_steps)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    return X_train, y_train, train_x ,test_x, train_y, test_y

#Function to calculate error and plot graph
def error_and_plot(model,train,n_steps,n_features, split_size):
    history = []
    for i in range(n_steps):
        history.append(array(train[(split_size-n_steps)+i]))
    
    predictions = []
    for t in range(len(test)):
        input_vec = np.zeros([n_steps,1])
        for i in range (n_steps):
            input_vec[i] = history[i][0]
        
        input_vec = input_vec.reshape((1,n_steps,n_features))
        yhat = model.predict(input_vec)
        predictions.append(yhat[0][0])
        history.append(test[t])
        history = history[-n_steps:]
        
    rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)
    # plot forecasts against actual outcomes
    pyplot.plot(test)
    pyplot.plot(predictions, color='red')
    pyplot.show()
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

df1 = df.loc[df['SOURCE_KEY_NUMBER'] == 1]
df1 = df1.reset_index()

#%%

#Creating X and y for training
n_steps = 8
n_features = 1
X = df1[['IRRADIATION']]
y = df1['AC_POWER']
X_train, y_train, train_x, test_x, train_y, test_y = creates_series(X, y, n_steps , n_features,1500)

#%%
#model definition
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=200)


input_vec = X_train[0].reshape((1,n_steps,n_features))

yhat = model.predict(input_vec)
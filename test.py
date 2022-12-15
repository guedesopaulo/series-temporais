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
from tensorflow.keras.optimizers import RMSprop
from math import sqrt
from sklearn.metrics import mean_squared_error
#%%
#Function to split one variable time series sequences
def split_sequence2(sequence_x1, sequence_x2, sequence_y, n_steps):
    X, y = list(), list()

    for i in range(len(sequence_x1)):
		# find the end of this pattern
        end_ix = i + n_steps
		# check if we are beyond the sequence
        if end_ix > len(sequence_x1)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = np.concatenate([sequence_x1[i:end_ix],sequence_x2[i:end_ix]]),  sequence_y[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def split_sequence(sequence_x1, sequence_y, n_steps):
    X, y = list(), list()

    for i in range(len(sequence_x1)):
		# find the end of this pattern
        end_ix = i + n_steps
		# check if we are beyond the sequence
        if end_ix > len(sequence_x1)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence_x1[i:end_ix], sequence_y[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

#Function to create series
def creates_series(series_x1, series_y, n_steps,split_size):
    X1 = series_x1.values
    Y = series_y.values
    train_x1, test_x1 = X1[0:split_size], X1[split_size:]
    train_y, test_y = Y[0:split_size], Y[split_size:]
    X_train, y_train = split_sequence(train_x1,train_y, n_steps)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    y_train = y_train.reshape((y_train.shape[0], 1))
    return X_train, y_train#, train_x ,test_x, train_y, test_y

def creates_series2(series_x1, series_x2, series_y, n_steps,split_size):
    X1 = series_x1.values
    X2 = series_x2.values
    Y = series_y.values
    train_x1, test_x1 = X1[0:split_size], X1[split_size:]
    train_x2, test_x2 = X2[0:split_size], X2[split_size:]
    train_y, test_y = Y[0:split_size], Y[split_size:]
    X_train, y_train = split_sequence(train_x1,train_x2,train_y, n_steps)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    y_train = y_train.reshape((y_train.shape[0], 1))
    return X_train, y_train#, train_x ,test_x, train_y, test_y




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
df["AC_POWER"]=(df["AC_POWER"]-df["AC_POWER"].mean())/df["AC_POWER"].std()
df['IRRADIATION']=(df['IRRADIATION']-df['IRRADIATION'].mean())/df['IRRADIATION'].std()
df1 = df.loc[df['SOURCE_KEY_NUMBER'] == 1]
df1 = df1.reset_index()

df2 = df.loc[df['SOURCE_KEY_NUMBER'] == 2]
df2 = df2.reset_index()

df3 = df.loc[df['SOURCE_KEY_NUMBER'] == 3]
df3 = df3.reset_index()

df4 = df.loc[df['SOURCE_KEY_NUMBER'] == 4]
df4 = df4.reset_index()

df5 = df.loc[df['SOURCE_KEY_NUMBER'] == 5]
df5 = df5.reset_index()

df6 = df.loc[df['SOURCE_KEY_NUMBER'] == 6]
df6 = df6.reset_index()

df7 = df.loc[df['SOURCE_KEY_NUMBER'] == 7]
df7 = df7.reset_index()

df8 = df.loc[df['SOURCE_KEY_NUMBER'] == 8]
df8 = df8.reset_index()

#%%

#Creating X and y for training
n_steps = 4
n_features = 1
X1 = df1['IRRADIATION']
X2 = df2['IRRADIATION']
X3 = df3['IRRADIATION']
X4 = df4['IRRADIATION']
X5 = df5['IRRADIATION']
X6 = df6['IRRADIATION']
X7 = df7['IRRADIATION']
X8 = df8['IRRADIATION']
y1 = df1['AC_POWER']
y2 = df2['AC_POWER']
y3 = df3['AC_POWER']
y4 = df4['AC_POWER']
y5 = df4['AC_POWER']
y6 = df4['AC_POWER']
y7 = df4['AC_POWER']
y8 = df4['AC_POWER']
X_train1, y_train1 = creates_series(X1,y1 ,n_steps,2070)
X_train2, y_train2 = creates_series(X2,y2 ,n_steps,2070)
X_train3, y_train3 = creates_series(X3,y3 ,n_steps,2070)
X_train4, y_train4 = creates_series(X4,y4 ,n_steps,2070)
X_train5, y_train5 = creates_series(X5,y5 ,n_steps,2070)
X_train6, y_train6 = creates_series(X6,y6 ,n_steps,2070)
X_train7, y_train7 = creates_series(X7,y7 ,n_steps,2070)
X_train8, y_train8 = creates_series(X8,y8 ,n_steps,2070)

X_train = np.concatenate((X_train1,X_train2,X_train3,X_train4,X_train5,X_train6,X_train7,X_train8))


y_train = np.concatenate((y_train1,y_train2,y_train3,y_train4,y_train5,y_train6,y_train7,y_train8))
#%%
#model definition
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(n_steps, 1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')


model.fit(X_train, y_train, epochs=5)

input_vec = X_train[0].reshape(1,4,1)


yhat = model.predict(input_vec)

#%%

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#%%
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

a = series_to_supervised(df1)
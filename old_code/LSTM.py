#Imports
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from pandas import read_csv
from matplotlib import pyplot
#%%

#Function definitions


#Function to split one variable time series sequences
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

#Function to create series
def creates_series(series, n_steps, n_features,split_size):
    X = series.values
    train, test = X[0:split_size], X[split_size:]
    X_train, y_train = split_sequence(train, n_steps)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    return X_train, y_train, train ,test

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
#VANNILA LSTM

#Loading data set
n_steps = 30
n_features = 1
series = read_csv('old_code/temp.csv', header=0, index_col=0)
X_train, y_train, train, test = creates_series(series, n_steps , n_features,3000)

#%%

#model definition
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=30)


#error analysis 
error_and_plot(model,train,n_steps,n_features, 3000)

#%%

#Stacked LSTM


# define model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=30)

#error analysis 
error_and_plot(model,train,n_steps,n_features, 3000)

#%%

#BIDIRECTIONAL LSTM

# define model
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=30)

#error analysis 
error_and_plot(model,train,n_steps,n_features, 3000)
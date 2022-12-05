from pandas import read_csv
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
from pandas import DataFrame
series = read_csv('temp.csv', header=0, index_col=0)


series.plot(figsize = (11, 7))
#%%
auto_cor = autocorrelation_plot(series)
auto_cor.plot(figsize = (11, 7))
pyplot.show()

#%%

#diff
diff = list()
X = series.values


for i in range(1, len(X)):
 	value = X[i] - X[i - 1]
 	diff.append(value)


pyplot.plot(diff)
pyplot.show()

a = series.drop(series.index[0])

for i in range(1, len(diff)):
    a["Temp"][i] = diff[i][0]

series = a
#%%
auto_cor = autocorrelation_plot(series)
auto_cor[0:10].plot(figsize = (11, 7))
pyplot.show()
#%%

# single exponential smoothing
from math import sqrt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error
# prepare data
X = series.values
train, test = X[0:3000], X[3000:]
history = [x for x in train]
predictions = list()
# create class
# fit model
# make prediction

# walk-forward validation
for t in range(len(test)):
	model = SimpleExpSmoothing(history)
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
#%%
"""
#using ARIMA
model = ARIMA(series, order=(5,1,0))
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())
# line plot of residuals
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
# density plot of residuals
residuals.plot(kind='kde')
pyplot.show()
# summary stats of residuals
print(residuals.describe())
"""
#%%

# evaluate an ARIMA model using a walk-forward validation
from sklearn.metrics import mean_squared_error
from math import sqrt
# split into train and test sets
X = series.values
#size = int(len(X) * 0.66)
train, test = X[0:3000], X[3000:]
history = [x for x in train]
predictions = list()
# walk-forward validation
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
#%%







#%%
"""
import matplotlib.pyplot as plt
#plt.style.use('fivethirtyeight')
#plt.rcParams['lines.linewidth'] = 1.5


# Modeling and Forecasting
# ==============================================================================
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster

data_train, data_test = series[0:3000], series[3000:]

print(f"Train dates : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})")
print(f"Test dates  : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")


data_train["Temp"].plot(use_index=True)
data_test["Temp"].plot(use_index=True)

#data_train["Temp"].plot(ax=ax, label='train')
#data_test["Temp"].plot(ax=ax, label='test')
#ax.legend();
plt.show()
#%%
#split a univariate sequence into samples
from numpy import array
from keras.models import Sequential
from keras.layers import Dense

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

n_steps = 365
#split into samples
X = data_train.values
X_test = data_test.values
X, y = split_sequence(X, n_steps)
X_test, y_test = split_sequence(X_test, n_steps)
# define model
model = Sequential()
model.add(Dense(1024, activation='relu', input_dim=n_steps))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=100, verbose=0)
# demonstrate prediction
series_pred = series
for i in range (2635):
    series_pred["Temp"][i] = model.predict(X)

#%%
from sklearn.neural_network import MLPClassifier 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster
import matplotlib.pyplot as plt
clf = MLPClassifier(hidden_layer_sizes=(8,32,64,1),activation="relu",random_state=1)
train, test = series[0:3000], series[3000:]

forecaster = ForecasterAutoreg(
                regressor = MLPRegressor(hidden_layer_sizes=(8,32,64,1)),
                lags      = 1
             )
forecaster.fit(y = train["Temp"])

pred = forecaster.predict(steps = 650)
pred.head(5)

fig, ax = plt.subplots(figsize=(9, 4))
test['Temp'].plot(ax=ax, label='test')
pred.plot(ax=ax, label='predictions')
ax.legend();

error_mse = mean_squared_error(
                y_true = test['Temp'],
                y_pred = pred
            )

print(f"Test error (mse): {error_mse}")
"""
#%%

from numpy import array
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
 
# split a univariate sequence into samples
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

X = series.values
train, test = X[0:3000], X[3000:]
n_steps = 30
X_train, y_train = split_sequence(train, n_steps)
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_steps))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=1000)

    
history = []
input_vec = np.zeros([30,1])

for i in range(n_steps):
    history.append(array(train[2970+i]))

predictions = []
for t in range(len(test)):
    
    yhat = model.predict(history[-30:])
    predictions.append(yhat)
    history.append(array(yhat))
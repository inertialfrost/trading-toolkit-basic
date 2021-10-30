import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Load data
company = 'TCS.NS'

start = dt.datetime(2020, 6, 6)
end = dt.datetime(2021, 1, 1)

data = yf.download(company, start, end)

# print(data)

# Prepare data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

d_returns = []

for x in range(1, len(data)):
    d_returns.append((data['Close'][x] - data['Close'][x-1])/data['Close'][x-1])

# print(d_returns)
dr_np = np.array(d_returns)

# print(len(dr_np))
print(np.shape(dr_np))

# print(scaled_data)

prediction_days = 20

# data used for training
x_train = []
# data against which the training is benchmarked and corrected
y_train = []

for x in range(prediction_days, len(scaled_data)):
    # x_train is a 2D array of len len(data) - 20 days
    # each row in x_train is the last 20 day prices for that row day
    # seems like for each day, we would want to train on the last 20 days
    x_train.append(scaled_data[x-prediction_days:x, 0])
    # y_train is simply the data[20:]
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# print('x_train shape: ')
# print(np.shape(x_train))
# print('y_train shape: ')
# print(np.shape(y_train))

# # Build the model
# Need to understand what are the other model types available
model = Sequential()

# Pretty straightforward, set up model with 50 units (nodes in a layer)
# The input is going to be one row for x_train, that is an array of 20
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# understand dropout, apparently sets a random number of units to 0
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

# overall 3 layers added
# # Prediction of the next closing price
# This is an output layer, called Dense, and hence one unit
model.add(Dense(units=1))

# set up the model, optimizers can be selected, loss function also selected
model.compile(optimizer='adam', loss='mean_squared_error')
# y_train is basically the output function that needs to be matched
# need to understand what epoch and batch_size mean
model.fit(x_train, y_train, epochs=25, batch_size=32)

# # Test model accuracy on existing data

# # Load test data
test_start = dt.datetime(2021, 1, 1)
test_end = dt.datetime.now()

# test_data = web.DataReader(company, 'yahoo', test_start, test_end)
# get different/newer data to test
test_data = yf.download(company, start, end)
actual_prices = test_data['Close'].values

# connected both datasets to get better image
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
# model_inputs seem to be the last bit of the total dataset, need to understand
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
# converting to 1D array to be able to plot
model_inputs = model_inputs.reshape(-1, 1)
# converting back to actual scale
model_inputs = scaler.transform(model_inputs)

# # Make predictions on test data

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

# continue onwards from here

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot test predictions
plt.plot(actual_prices, color='black', label=f'Actual {company} Price')
plt.plot(predicted_prices, color='green', label=f'Predicted {company} Price')
plt.title(f'{company} Share Price')
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()

# Predict next day

real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs ) + 1, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f'Prediction: {prediction}')
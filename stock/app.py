import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import pandas_datareader as data
from keras.models import load_model
import streamlit as st

import yfinance as yf

start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
stock_data = yf.download(user_input, start, end)

# Describing data
st.subheader('Data from 2010 - 2019')
st.write(stock_data.describe())

# visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(stock_data.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = stock_data.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(stock_data.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = stock_data.Close.rolling(100).mean()
ma200 = stock_data.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(stock_data.Close, 'b')
st.pyplot(fig)

# splitting data into training and testing
data_training = pd.DataFrame(stock_data['Close'][0:int(len(stock_data) * 0.70)])
data_testing = pd.DataFrame(stock_data['Close'][int(len(stock_data) * 0.70):int(len(stock_data))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)

# splitting data into x_train and y_train
# x_train = []
# y_train = []
#
# for i in range(100, data_training_array.shape[0]):
#     x_train.append(data_training_array[i - 100: i])
#     y_train.append(data_training_array[i, 0])
#
# x_train, y_train = np.array(x_train), np.array(y_train)

# load model
model = load_model('/Users/sadafrizwan/Desktop/stock-prediction/stock/keras_model.h5')
# model = load_model('my_model.keras')




# testing part
past_100_days = data_training.tail(100)
final_stock_data = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_stock_data)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100: i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# final graph
st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original price')
plt.plot(y_predicted, 'r', label='Predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


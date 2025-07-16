# crypto_predictor.py

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from ta.momentum import RSIIndicator
from ta.trend import MACD

# ---------------------
# Step 1: Load Data
# ---------------------
btc = yf.download('BTC-USD', start='2020-01-01', end='2025-07-15')

# ---------------------
# Step 2: Add Indicators
# ---------------------
def add_indicators(df):
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df.dropna(inplace=True)
    return df

btc = add_indicators(btc)

# ---------------------
# Step 3: Prepare Sequences
# ---------------------
features = ['Close', 'MA7', 'RSI', 'MACD']
data = btc[features].values

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i][0])
    return np.array(X), np.array(y)

seq_len = 60
X, y = create_sequences(data_scaled, seq_len)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ---------------------
# Step 4: Build LSTM Model
# ---------------------
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=25, batch_size=32, vali_

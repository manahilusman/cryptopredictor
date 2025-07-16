# cryptopredictor
import yfinance as yf
btc = yf.download('BTC-USD', start='2020-01-01', end='2025-07-15')
from ta.momentum import RSIIndicator
from ta.trend import MACD

def add_indicators(df):
    df['MA7'] = df['Close'].rolling(window=7).mean()  # 7-day moving average
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df.dropna(inplace=True)
    return df

from sklearn.preprocessing import MinMaxScaler
features = ['Close', 'MA7', 'RSI', 'MACD']
scaler = MinMaxScaler()
data = scaler.fit_transform(btc[features])

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])  # previous n days
        y.append(data[i][0])            # predict Close price only
    return np.array(X), np.array(y)

X, y = create_sequences(data, 60)

split = int(0.8 * len(X))  # 80% for training
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))  # Prevent overfitting
model.add(LSTM(100))     # Second layer, returns only the final output
model.add(Dropout(0.2))
model.add(Dense(1))      # Predict single value (close price)

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)

# Rebuild full row for inverse transform (we only predicted Close)
y_test_inv = scaler.inverse_transform(np.hstack((y_test.reshape(-1,1),
                                                  np.zeros((len(y_test), 3)))))

y_pred_inv = scaler.inverse_transform(np.hstack((y_pred,
                                                  np.zeros((len(y_pred), 3)))))

import matplotlib.pyplot as plt
plt.figure(figsize=(14,6))
plt.plot(y_test_inv[:,0], label='Actual Price')
plt.plot(y_pred_inv[:,0], label='Predicted Price')
plt.title("BTC Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()

import streamlit as st
st.title("Crypto Price Predictor")
st.write("This app will show predicted vs actual BTC prices based on LSTM")

st.line_chart(y_pred_inv[:100, 0])  # Show first 100 predictions

streamlit run app.py

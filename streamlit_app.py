# streamlit_app.py

import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Title and Introduction
st.title("🪙 BTC Price Predictor Dashboard")
st.write("""
This interactive dashboard shows predictions of Bitcoin prices using an LSTM model trained on historical data + technical indicators.

🔹 The model used: 2-layer LSTM  
🔹 Features: Close Price, MA7, RSI, MACD  
🔹 Forecast: Next-day price prediction  
""")

# Show the saved prediction plot
st.subheader("📈 Actual vs Predicted BTC Price")
try:
    image = Image.open('btc_prediction.png')
    st.image(image, caption="Model Prediction vs Real BTC Price", use_column_width=True)
except:
    st.warning("⚠️ Prediction plot not found. Please run `crypto_predictor.py` first to generate the model and plot.")

# Add credits or further explanation
st.markdown("""
---

**How the model works**:
- It uses the past 60 days of data to predict the next day's price.
- Trained on ~4 years of BTC data from Yahoo Finance.
- Uses technical analysis indicators like Moving Averages, RSI, and MACD.

**Try This**:
- Modify `crypto_predictor.py` to also use Ethereum (ETH).
- Add a multi-crypto toggle here.
- Deploy this app to [Streamlit Cloud](https://streamlit.io/cloud) to make it public.

---
Made with ❤️ using Python and Streamlit.
""")

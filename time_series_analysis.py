import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Binance API から仮想通貨の価格データを取得
def get_binance_data(symbol="BTCUSDT", interval="1h", limit=500):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df
    else:
        raise ValueError(f"Binance API からデータ取得失敗: {response.status_code}")

# Binance API からデータ取得
df = get_binance_data()

# 📌 価格データの可視化
plt.figure(figsize=(12, 5))
plt.plot(df["close"], label="BTCUSDT Price", color="blue")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.title("Bitcoin Price Trend")
plt.show()

# 📌 GARCH（ボラティリティ予測）
returns = df["close"].pct_change().dropna()
garch_model = arch_model(returns, vol="Garch", p=1, q=1)
garch_result = garch_model.fit(disp="off")
forecast = garch_result.forecast(start=len(returns), horizon=10)
predicted_volatility = forecast.variance.values[-1] ** 0.5

# 📌 ARIMA（短期価格予測）
arima_model = ARIMA(df["close"], order=(5, 1, 0))
arima_result = arima_model.fit()
forecast_price = arima_result.forecast(steps=10)

# 📌 LSTM（長期トレンド予測）
lookback = 60
X, y = [], []
for i in range(lookback, len(df)):
    X.append(df["close"].values[i-lookback:i])
    y.append(df["close"].values[i])
X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=5, batch_size=16, verbose=0)

# 📌 予測
last_60 = df["close"].values[-lookback:].reshape(1, lookback, 1)
lstm_predicted_price = model.predict(last_60)[0][0]

# 📌 結果を出力
print(f"GARCH 予測ボラティリティ: {predicted_volatility:.6f}")
print(f"ARIMA 予測価格: {forecast_price.iloc[-1]:.2f}")
print(f"LSTM 予測価格: {lstm_predicted_price:.2f}")

# 📌 予測結果のプロット
plt.figure(figsize=(12, 5))
plt.plot(df.index[-100:], df["close"].values[-100:], label="Real Price", color="blue")
plt.axhline(y=forecast_price.iloc[-1], color="red", linestyle="dashed", label="ARIMA Prediction")
plt.axhline(y=lstm_predicted_price, color="green", linestyle="dashed", label="LSTM Prediction")
plt.legend()
plt.title("BTC Price Predictions")
plt.show()

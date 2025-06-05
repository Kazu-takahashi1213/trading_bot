import os
import pandas as pd
import numpy as np
import requests
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

load_dotenv()
API_KEY = os.getenv('BYBIT_API_KEY')
API_SECRET = os.getenv('BYBIT_API_SECRET')

session = HTTP(testnet=True, api_key=API_KEY, api_secret=API_SECRET)


def get_binance_data(symbol='BTCUSDT', interval='1h', limit=500):
    url = 'https://api.binance.com/api/v3/klines'
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume','close_time','quote','trades','taker_base','taker_quote','ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open','high','low','close','volume']].astype(float)
    return df


def compute_signals(df, short_window=9, long_window=20):
    df['SMA_short'] = df['close'].rolling(short_window).mean()
    df['SMA_long'] = df['close'].rolling(long_window).mean()
    df['Signal'] = 0
    df.loc[df['SMA_short'] > df['SMA_long'], 'Signal'] = 1
    df.loc[df['SMA_short'] < df['SMA_long'], 'Signal'] = -1
    return df.dropna()


def train_arima(series):
    model = ARIMA(series, order=(5, 1, 0))
    return model.fit()


def forecast_next(model):
    forecast = model.forecast(steps=1)
    return float(forecast.iloc[0])


def train_lstm(series, lookback=60, epochs=10):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i])

    X = np.array(X)
    y = np.array(y)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
    return model, scaler


def forecast_next_lstm(model, series, scaler, lookback=60):
    seq = series.values[-lookback:]
    seq = scaler.transform(seq.reshape(-1, 1))
    seq = seq.reshape(1, lookback, 1)
    pred = model.predict(seq, verbose=0)
    return float(scaler.inverse_transform(pred)[0, 0])


if __name__ == '__main__':
    df = get_binance_data()
    df = compute_signals(df)

    arima_model = train_arima(df['close'])
    next_arima = forecast_next(arima_model)

    lstm_model, scaler = train_lstm(df['close'])
    next_lstm = forecast_next_lstm(lstm_model, df['close'], scaler)

    next_price = (next_arima + next_lstm) / 2
    last_price = df['close'].iloc[-1]
    predicted_return = (next_price - last_price) / last_price
    last_signal = df['Signal'].iloc[-1]

    print(f'Last close price: {last_price}')
    print(f'ARIMA predicted price: {next_arima}')
    print(f'LSTM predicted price: {next_lstm}')
    print(f'Ensemble predicted price: {next_price}')
    print(f'Predicted return: {predicted_return:.6f}')
    print(f'Last golden cross signal: {"BUY" if last_signal==1 else "SELL" if last_signal==-1 else "HOLD"}')

    if last_signal == 1 and predicted_return > 0:
        print('Executing buy order.')
        session.place_order(symbol='BTCUSDT', side='Buy', qty=0.01, order_type='Market')
    elif last_signal == -1 and predicted_return < 0:
        print('Executing sell order.')
        session.place_order(symbol='BTCUSDT', side='Sell', qty=0.01, order_type='Market')
    else:
        print('No trade executed.')

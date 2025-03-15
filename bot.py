!pip install requests pandas numpy matplotlib pybit xgboost scikit-learn backtesting

import os
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

# Bybit APIキーの設定
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")

# APIクライアントの作成
session = HTTP(
    testnet=True,  # 本番環境の場合は False
    api_key=API_KEY,
    api_secret=API_SECRET
)

import requests
import pandas as pd
import time

def get_historical_data(symbol="BTCUSDT", interval=60, limit=1000):
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    
    response = requests.get(url, params=params)
    data = response.json()

    if "result" in data:
        df = pd.DataFrame(data["result"]["list"], columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)
        return df
    else:
        raise ValueError("APIからデータ取得失敗")

# 例: 1時間足データを取得
df = get_historical_data(symbol="BTCUSDT", interval=60, limit=1000)
print(df.head())

def calculate_moving_averages(df, short_window=9, long_window=20):
    df["SMA_Short"] = df["close"].rolling(window=short_window).mean()
    df["SMA_Long"] = df["close"].rolling(window=long_window).mean()
    
    # ゴールデンクロス / デッドクロス判定
    df["Signal"] = 0
    df.loc[df["SMA_Short"] > df["SMA_Long"], "Signal"] = 1
    df.loc[df["SMA_Short"] < df["SMA_Long"], "Signal"] = -1
    
    return df

df = calculate_moving_averages(df)
print(df.tail())

from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class MovingAverageCrossStrategy(Strategy):
    short_window = 9
    long_window = 20

    def init(self):
        self.sma_short = self.I(lambda x: x.rolling(self.short_window).mean(), self.data.Close)
        self.sma_long = self.I(lambda x: x.rolling(self.long_window).mean(), self.data.Close)

    def next(self):
        if crossover(self.sma_short, self.sma_long):
            self.buy()
        elif crossover(self.sma_long, self.sma_short):
            self.sell()

bt = Backtest(df, MovingAverageCrossStrategy, cash=10000, commission=.002)
results = bt.run()
bt.plot()

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_features(df):
    df["return_5"] = df["close"].pct_change(5)
    df["volatility"] = df["close"].rolling(10).std()
    df["SMA_Diff"] = df["SMA_Short"] - df["SMA_Long"]
    
    df.dropna(inplace=True)
    return df

df = prepare_features(df)

X = df[["return_5", "volatility", "SMA_Diff"]]
y = df["Signal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"モデルの精度: {accuracy:.2f}")

from pybit.unified_trading import WebSocket

ws = WebSocket(testnet=True, channel_type="linear")

def handle_message(msg):
    close_price = float(msg["data"]["c"])
    
    # 最新価格をデータフレームに追加
    global df
    df.loc[pd.Timestamp.now()] = [None, None, None, close_price, None, None]
    
    # 移動平均を計算
    df = calculate_moving_averages(df)
    
    # 最新のシグナルを取得
    last_signal = df["Signal"].iloc[-1]
    
    if last_signal == 1:
        print("ゴールデンクロス！買い注文を発注")
        session.place_order(symbol="BTCUSDT", side="Buy", qty=0.01, order_type="Market")
    elif last_signal == -1:
        print("デッドクロス！売り注文を発注")
        session.place_order(symbol="BTCUSDT", side="Sell", qty=0.01, order_type="Market")

ws.kline_stream(interval=1, symbol="BTCUSDT", callback=handle_message)

def apply_risk_management(df, stop_loss_pct=0.05):
    atr = df["high"] - df["low"]
    df["Stop_Loss"] = df["close"] * (1 - stop_loss_pct)  # 5%のストップロス
    df["ATR"] = atr.rolling(14).mean()
    return df

df = apply_risk_management(df)


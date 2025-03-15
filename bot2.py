import os
import pandas as pd
import numpy as np
import time
import requests
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 環境変数の読み込み
load_dotenv()

# Bybit APIキーの取得（安全な管理）
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")

if not API_KEY or not API_SECRET:
    raise ValueError("APIキーが設定されていません。.env ファイルを確認してください。")

# APIクライアントの作成
session = HTTP(testnet=True, api_key=API_KEY, api_secret=API_SECRET)

# 過去データ取得（ボラティリティベースのバー用）
def get_historical_data(symbol="BTCUSDT", interval=60, limit=1000):
    url = "https://api.bybit.com/v5/market/kline"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
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

def prepare_features(df):
    df["return_5"] = df["close"].pct_change(5)
    df["volatility"] = df["close"].rolling(10).std()
    df["SMA_Diff"] = df["close"].rolling(9).mean() - df["close"].rolling(20).mean()
    df.dropna(inplace=True)
    return df

def apply_triple_barrier(df, pt=0.02, sl=0.02, max_holding=5):
    df["pt"] = df["close"] * (1 + pt)
    df["sl"] = df["close"] * (1 - sl)
    df["t1"] = df.index + pd.Timedelta(minutes=max_holding)
    return df

# メイン処理
def main():
    df = get_historical_data()
    df = prepare_features(df)
    df = apply_triple_barrier(df)

    # XGBoostモデルの学習
    X = df[["return_5", "volatility", "SMA_Diff"]]
    y = np.sign(df["return_5"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"モデルの精度: {accuracy:.2f}")

    # バックテスト用のストラテジー
    class XGBoostStrategy(Strategy):
        def init(self):
            self.model = model
            self.data_features = pd.DataFrame({
                "return_5": self.data.Close.pct_change(5),
                "volatility": self.data.Close.rolling(10).std(),
                "SMA_Diff": self.data.Close.rolling(9).mean() - self.data.Close.rolling(20).mean()
            }).dropna()

        def next(self):
            if len(self.data_features) >= len(self.data.Close):
                features = self.data_features.iloc[-1:].values
                prediction = self.model.predict(features)
                if prediction == 1:
                    self.buy()
                elif prediction == -1:
                    self.sell()

    bt = Backtest(df, XGBoostStrategy, cash=10000, commission=.002)
    results = bt.run()
    bt.plot()

if __name__ == "__main__":
    main()

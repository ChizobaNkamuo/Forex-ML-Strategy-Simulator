import ta
import numpy as np

def add_indicators(data):
    data["rsi"] = ta.momentum.RSIIndicator(data["Close"]).rsi()
    data["ema_10"] = ta.trend.EMAIndicator(data["Close"], window=10).ema_indicator()
    data["macd"] = ta.trend.MACD(data["Close"]).macd()
    data.dropna(inplace=True)

    return data 

def standardise(data):
    for feature in data:
        data[feature] = (data[feature] - data[feature].mean()) / data[feature].std()


def create_flattened_features(data, window=10):
    features, targets = [], []
    for i in range(len(data) - window):
        past_window = data.iloc[i:i+window].values.flatten()
        close_value = data.iloc[i+window]["Close"]
        open_value  = data.iloc[i+window]["Open"]
        features.append(past_window)
        targets.append(1 if close_value >= open_value else 0)
    return np.array(features), np.array(targets)


import ta
import numpy as np
import ta.volatility
import pandas as pd
from sklearn.preprocessing import StandardScaler
SPLIT_PERCENTAGE = 0.8

def add_indicators(data):
    macd = ta.trend.MACD(data["Close"]) # Mean average convergence divergence
    bb = ta.volatility.BollingerBands(close=data["Close"], window=20)
    stochastic = ta.momentum.StochasticOscillator(data["High"], data["Low"], data["Close"])
    date = pd.to_datetime(data["date"])

    data["Change"] = data["Close"] - data["Open"]
    for lag in range(1, 5):
        data[f"Change_lag_{lag}"] = data["Change"].shift(lag)
        
    data["MA"] = data["Close"].rolling(10).mean() # Moving average
    data["MACD"] =  macd.macd()
    data["MACD_histogram"] = macd.macd_diff()
    data["RSI"] = ta.momentum.RSIIndicator(data["Close"], window = 10).rsi() # Relative strength index
    data["ROC"] = ta.momentum.roc(data["Close"], window = 2)
    data["CCI"] = ta.trend.cci(data["High"], data["Low"], data["Close"])
    data["bb_high"] = bb.bollinger_hband()
    data["bb_low"] = bb.bollinger_lband()
    data["bb_pband"] = bb.bollinger_pband()
    data["bb_width"] = bb.bollinger_wband()
    data["stochastic"] = stochastic.stoch()
    data["stochastic_signal"] = stochastic.stoch_signal()
    data["day_sin"] = np.sin(2 * np.pi * date.dt.weekday / 7)
    data["day_cos"] = np.cos(2 * np.pi * date.dt.weekday / 7)
    
    data.dropna(inplace=True)
    return data 

def create_macro_indicators(data, macro_indicators):
    data["date"] = pd.to_datetime(data["date"])

    for indicator in macro_indicators:
        indicator_data = pd.read_csv(f"./data/{indicator}.csv")
        indicator_data["observation_date"] = pd.to_datetime(indicator_data["observation_date"])
        indicator_data = indicator_data.rename(columns={"observation_date": "date"})
        data = pd.merge_asof(
            data.sort_values("date"),
            indicator_data.sort_values("date"),
            on="date",
            direction="backward"
        )
    return data

def split_data(data, feature_columns, target_columns, sequence_length, scale = False):
    split = int(SPLIT_PERCENTAGE * len(data))
    train_data = add_indicators(data[:split].copy())
    test_data = add_indicators(data[split:].copy())
    
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_data[feature_columns]) if scale else train_data[feature_columns]
    test_features_scaled = scaler.transform(test_data[feature_columns]) if scale else test_data[feature_columns]

    X_train, y_train = create_sequences(train_features_scaled, train_data[target_columns].values, sequence_length)
    X_test, y_test = create_sequences(test_features_scaled, test_data[target_columns].values, sequence_length)
    
    return X_test, X_train, y_test, y_train, scaler

def create_sequences(features, targets, sequence_length):
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i+sequence_length])
        y.append(targets[i+sequence_length])
    return np.array(X), np.array(y)

import ta, os
import numpy as np
import ta.volatility
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
SPLIT_PERCENTAGE = 0.8
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def add_indicators(data):
    """
    Add technical indicators to data
    """
    macd = ta.trend.MACD(data["Close"]) # Mean average convergence divergence
    bb = ta.volatility.BollingerBands(close=data["Close"], window=20)

    data["date"] = pd.to_datetime(data["date"])
    data["MA"] = data["Close"].rolling(10).mean() # Moving average
    data["MACD"] =  macd.macd()
    data["MACD_Histogram"] = macd.macd_diff()
    data["ROC"] = ta.momentum.roc(data["Close"])
    data["Momentum"] = data["Close"] - data["Close"].shift(10)
    data["RSI"] = ta.momentum.RSIIndicator(data["Close"], window = 10).rsi() # Relative strength index
    data["BB_Pband"] = bb.bollinger_pband()
    data["BB_Width"] = bb.bollinger_wband()
    data["CCI"] = ta.trend.cci(data["High"], data["Low"], data["Close"])
 
    data.dropna(inplace=True)

    return data, data["date"]

def create_macro_indicators(data, macro_indicators):
    """
    Merge macroeconomic indicators with data backwards to ensure that data is only avaliable on the dates it would've been avaliable
    """
    data["date"] = pd.to_datetime(data["date"], format = "%d/%m/%Y")
    
    for indicator in macro_indicators:
        indicator_data = pd.read_csv(os.path.join(BASE_DIR, f"data/{indicator}.csv"))
        indicator_data["date"] = pd.to_datetime(indicator_data["date"], format = "mixed")
        data = pd.merge_asof(
            data.sort_values("date"),
            indicator_data.sort_values("date"),
            on="date",
            direction="backward"
        )
    data.dropna(inplace=True)
    return data

def split_data(data, feature_columns, target_columns, sequence_length, scaler_name):
    """
    Split data into test and train and then format it for the model to use
    """
    split = int(SPLIT_PERCENTAGE * len(data))
    train_data, train_date = add_indicators(data[:split].copy())
    test_data, test_date = add_indicators(data[split:].copy())
    
    scaler = MinMaxScaler()
    
    train_features_scaled = scaler.fit_transform(train_data[feature_columns])
    test_features_scaled = scaler.transform(test_data[feature_columns])
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"{scaler_name}.gz"))

    X_train, y_train = create_sequences(train_features_scaled, train_data[target_columns].values, sequence_length)
    X_test, y_test = create_sequences(test_features_scaled, test_data[target_columns].values, sequence_length)
    
    return X_test, X_train, y_test, y_train, test_date[sequence_length:]

def create_sequences(features, targets, sequence_length):
    """
    Create sequences using the previous sequence_length features to predict the target value at sequence_length + 1
    """
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i+sequence_length])
        y.append(targets[i+sequence_length])
    return np.array(X), np.array(y)

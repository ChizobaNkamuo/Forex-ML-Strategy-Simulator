import ta
import numpy as np
import ta.volatility
from sklearn.preprocessing import StandardScaler

def add_indicators(data):
    close_column = data["Close"]
    data["ema"] = close_column.ewm(span = 5).mean().shift(1) # Exponential moving average
    data["log_return"] = np.log(close_column / close_column.shift(1))

    macd = ta.trend.MACD(close_column) # Mean average convergence divergence
    data["macd"] =  macd.macd().shift(1)
    data["macd_histogram"] = macd.macd_diff().shift(1)
    data["rsi"] = ta.momentum.RSIIndicator(close_column).rsi().shift(1) # Relative strength index

    bollinger_bands = ta.volatility.BollingerBands(close=close_column, window=20, window_dev=2)
    data["bb_high"] = bollinger_bands.bollinger_hband().shift(1)
    data["bb_low"] = bollinger_bands.bollinger_lband().shift(1)
    data["bb_pband"] = bollinger_bands.bollinger_pband().shift(1)
    data["bb_width"] = bollinger_bands.bollinger_wband().shift(1)

    stochastic = ta.momentum.StochasticOscillator(data["High"], data["Low"], close_column)
    data["stochastic"] = stochastic.stoch().shift(1)
    data["stochastic_signal"] = stochastic.stoch_signal().shift(1)

    data["atr"] = ta.volatility.AverageTrueRange(data["High"], data["Low"], close_column).average_true_range().shift(1)
    data["day"] = np.arange(len(close_column)) % 7
    
    data["close_shifted"] = data["Close"].shift(1)
    data["open_shifted"] = data["Open"].shift(1)
    
    data.dropna(inplace=True)
    return data 

def standardise(features, targets):
    scaler = StandardScaler()
    split = int(0.8 * len(features))
    features_train = scaler.fit_transform(features[:split])
    features_test = scaler.transform(features[split:])
    targets_train, targets_test = targets[:split], targets[split:]
    
    return features_train, features_test, targets_train, targets_test

def create_sequences(features_train, features_test, targets_train, targets_test, sequence_length=10):
    sequence_train, sequence_test = [], []
    
    for i in range(sequence_length, len(features_train)):
        sequence_train.append(features_train[i-sequence_length:i])

    for i in range(sequence_length, len(features_test)):
        sequence_test.append(features_test[i-sequence_length:i])
    
    return np.array(sequence_train), np.array(sequence_test), targets_train[sequence_length:].to_numpy(), targets_test[sequence_length:].to_numpy()

"""
def create_sequences(data, features_cols, target_col, sequence_length=10):
    X, y = [], []
    
    for i in range(sequence_length, len(data)):
        X.append(data[features_cols].iloc[i-sequence_length:i].values)
        y.append(data[target_col].iloc[i])
    return np.array(X), np.array(y)
"""
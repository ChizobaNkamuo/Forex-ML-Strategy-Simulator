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

    data["close_shifted"] = data["Close"].shift(1)
    data["open_shifted"] = data["Open"].shift(1)
    
    data.dropna(inplace=True)

    return data 

def standardise(data):
    return StandardScaler().fit_transform(data)
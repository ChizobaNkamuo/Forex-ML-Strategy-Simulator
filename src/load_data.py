import pandas as pd
import numpy as np
from src.feature_engineering import create_macro_indicators
macro_indicators = ["Fed_Funds_Rate","USA_CPI","EU_HICP","DAX_Close","EU_Interest","German_Interest","SP500"]
features_columns_tech = ["Close", "MA", "MACD", "MACD_Histogram", "BB_Width", "BB_Pband", "RSI","ROC","CCI", "Momentum"]
feature_columns_macro = ["Close"]
feature_columns_macro.extend(macro_indicators)
target_column = ["Change"]

def load():
    data = pd.read_csv("./data/fx_daily_EUR_USD.csv").sort_values(by=["date"])
    data = create_macro_indicators(data, macro_indicators)
    diff = (data["Close"] - data["Close"].shift(1))
    #print(threshold.calculate_threshold(diff))
    final_threshold = 0.0020400000000000045
    conditions = [
        diff > final_threshold,
        diff < -final_threshold
    ]

    data["Change"] = np.select(conditions, [2, 1], 0)
    data["Diff"] = diff
    data.dropna(inplace=True)

    data = data.reset_index()
    return data

def get_features_and_targets():
    return target_column, feature_columns_macro, features_columns_tech
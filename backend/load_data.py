import pandas as pd
import numpy as np
import feature_engineering, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

macro_indicators = ["Fed_Funds_Rate","USA_CPI","EU_HICP","DAX_Close","EU_Interest","German_Interest","SP500"]
features_columns_tech = ["Close", "MA", "MACD", "MACD_Histogram", "BB_Width", "BB_Pband", "RSI","ROC","CCI", "Momentum"]
feature_columns_macro = ["Close"]
feature_columns_macro.extend(macro_indicators)
target_column = ["Change"]


def load():
    """
    Load data, add macroeconomic indicators and classify price movements based on the calculated threshold
    """
    data = pd.read_csv(os.path.join(DATA_DIR, "fx_daily_EUR_USD.csv")).sort_values(by=["date"])
    data = feature_engineering.create_macro_indicators(data, macro_indicators)
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
    """
    Return the feature and target information
    """
    return target_column, feature_columns_macro, features_columns_tech
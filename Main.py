import pandas as pd
import numpy as np
import src.feature_engineering as feature_engineering
import src.feature_selection as feature_selection
import src.evaluate as evaluate
import src.model_components as model_components

macro_indicators = ["CPI","IORB","PPI","TRADE_BALANCE","GDP"] # "NON_FARM_EMPLOYEES","UNEMPLOYMENT",
features_columns = ["close_shifted", "high_shifted", "low_shifted", "open_shifted", "macd", "macd_histogram", "bb_width", "bb_pband", "rsi","stochastic","atr","day"] #"bb_high", "bb_low","ema", "stochastic_signal""
features_columns.extend(macro_indicators)
raw_data = pd.read_csv("./data/EURUSD2021-07-29-to-2025-07-01-1d.csv")
data = feature_engineering.add_indicators(raw_data, macro_indicators)

sequence_length = 50
features_train, features_test, targets_train, targets_test = feature_engineering.standardise(data[features_columns], data["log_return"])
feature_selection.heat_map(features_train, features_columns)
features_train, features_test, targets_train, targets_test = feature_engineering.create_sequences(features_train, features_test, targets_train, targets_test, sequence_length)

model, history = model_components.create(sequence_length, features_columns, features_train, targets_train)

predictions = model.predict(features_test)

evaluate.display_metrics(predictions, targets_test, raw_data, sequence_length)
evaluate.plot_results(predictions, targets_test)
evaluate.plot_training_history(history)
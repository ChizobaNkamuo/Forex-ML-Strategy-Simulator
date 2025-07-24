import pandas as pd
import numpy as np
import src.feature_engineering as feature_engineering
import src.feature_selection as feature_selection
import src.evaluate as evaluate
import src.model_components as model_components

features_columns = ["close_shifted", "open_shifted", "macd_histogram", "bb_width", "macd", "bb_pband", "bb_high", "bb_low","rsi","ema", "stochastic","stochastic_signal","atr","day"] #["close_shifted", "open_shifted", "ema", "macd", "macd_histogram", "rsi", "bb_pband", "bb_width", "bb_high", "bb_low"] # 
raw_data = pd.read_csv("./data/EURUSD2020-2025.csv")
data = feature_engineering.add_indicators(raw_data)

sequence_length = 50
features_train, features_test, targets_train, targets_test = feature_engineering.standardise(data[features_columns], data["log_return"])
features_train, features_test, targets_train, targets_test = feature_engineering.create_sequences(features_train, features_test, targets_train, targets_test, sequence_length)

model, history = model_components.create(sequence_length, features_columns, features_train, targets_train)

predictions = model.predict(features_test)

evaluate.display_metrics(predictions, targets_test, raw_data, sequence_length)
evaluate.plot_results(predictions, targets_test)
evaluate.plot_training_history(history)
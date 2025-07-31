import pandas as pd
import numpy as np
import src.feature_engineering as feature_engineering
import src.feature_selection as feature_selection
import src.evaluate as evaluate
import src.model_components as model_components

macro_indicators = ["CPI","IORB","PPI","TRADE_BALANCE","GDP"] # "NON_FARM_EMPLOYEES","UNEMPLOYMENT",
target_columns = ["Open", "Close", "High", "Low"]
feature_columns = target_columns[:]
target_columns_ti = ["Change"]
features_columns_ti = ["Change","Change_lag_1","Change_lag_2","Change_lag_3","Change_lag_4","Open", "Close", "High", "Low", "MACD", "MACD_histogram", "bb_width", "bb_pband", "RSI","ROC","stochastic","CCI","MA","day_sin","day_cos"] #"bb_high", "bb_low","ema", "stochastic_signal""
#features_columns_ti.extend(macro_indicators)
data = pd.read_csv("./data/EURUSD2021-07-29-to-2025-07-01-1d.csv")

sequence_length = 1
sequence_length_ti = 50
#feature_selection.heat_map(features_train, features_columns)
#data = feature_engineering.create_macro_indicators(data, macro_indicators) 
#data_ti = feature_engineering.create_macro_indicators(data.copy(deep = True), macro_indicators) 
features_test, features_train, targets_test, targets_train, _ = feature_engineering.split_data(data.copy(deep = True), feature_columns, target_columns, sequence_length)
features_test_ti, features_train_ti, targets_test_ti, targets_train_ti, scaler = feature_engineering.split_data(data, features_columns_ti, target_columns_ti, sequence_length_ti, scale = True)

#candle_stick_model, candle_stick_history = model_components.create_candle_stick_model(sequence_length, feature_columns, features_train, targets_train)
#predictions = candle_stick_model.predict(features_test)
technical_model, technical_history = model_components.create_indicator_model(sequence_length_ti, features_columns_ti, features_train_ti, targets_train_ti)
technical_predictions = technical_model.predict(features_test_ti)
evaluate.directional_accuracy(technical_predictions, targets_test_ti)

#predictions = target_scaler.inverse_transform(predictions)
#targets_test = target_scaler.inverse_transform(targets_test)
#training_dates = data["date"][len(data)-len(targets_test):]

#evaluate.plot_training_history(training_dates, predictions)
#evaluate.plot_candle_sticks(predictions, targets_test, training_dates)

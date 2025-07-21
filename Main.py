import pandas as pd
import src.feature_engineering as feature_engineering
import src.feature_selection as feature_selection
import src.evaluate as evaluate
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention, Concatenate
from keras import regularizers

features_columns = ["close_shifted", "open_shifted", "macd_histogram", "bb_width", "macd", "bb_pband", "bb_high", "bb_low","rsi","ema", "stochastic","stochastic_signal","atr","day"] #["close_shifted", "open_shifted", "ema", "macd", "macd_histogram", "rsi", "bb_pband", "bb_width", "bb_high", "bb_low"] # 
data = feature_engineering.add_indicators(pd.read_csv("./data/EURUSD2020-2025.csv"))

sequence_length = 50
features_train, features_test, targets_train, targets_test = feature_engineering.standardise(data[features_columns], data["log_return"])
features_train, features_test, targets_train, targets_test = feature_engineering.create_sequences(features_train, features_test, targets_train, targets_test, sequence_length)

regularisation = regularizers.l1_l2(l1=1e-5, l2=1e-4)
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True, input_shape=(sequence_length, len(features_columns)), kernel_regularizer=regularisation)),
    Dropout(0.2),
    Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=regularisation)),
    Dropout(0.2),
    Dense(25, kernel_regularizer=regularisation),
    Dense(1, kernel_regularizer=regularisation)
])

model.compile(optimizer="adam", loss="mae", metrics=[])
model.fit(features_train, targets_train, 
                   epochs=50, 
                   batch_size=32, 
                   validation_split=0.2,
                   verbose=1)

predictions = model.predict(features_test)

evaluate.display_metrics(predictions, targets_test)
evaluate.plot_results(predictions, targets_test)

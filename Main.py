import pandas as pd
import src.feature_engineering as feature_engineering
import src.model as model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

features_columns = ["close_shifted", "open_shifted", "ema", "macd", "macd_histogram", "rsi", "bb_pband", "bb_width", "bb_high", "bb_low"]
data = feature_engineering.add_indicators(pd.read_csv("./data/EURUSD2020-2025.csv"))
data[features_columns] = feature_engineering.standardise(data[features_columns])
features, targets = data[features_columns], data["log_return"]

split = int(0.8 * len(features))
features_train, features_test = features[:split], features[split:]
targets_train, targets_test = targets[:split], targets[split:]

predictor = LinearRegression()
predictor.fit(features_train, targets_train)
predictions = predictor.predict(features_test)
print(r2_score(targets_test, predictions))
print(mean_squared_error(targets_test, predictions))


print(targets.describe())
print(f"Target std: {targets.std()}")
for col in features_columns:
    corr = data[col].corr(data["log_return"])
    print(f"{col}: {corr:.4f}")

model.permutation_test(predictor, features_columns, features_test, targets_test)

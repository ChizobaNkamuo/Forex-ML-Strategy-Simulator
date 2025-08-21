import pandas as pd
import numpy as np
import src.feature_engineering as feature_engineering
import src.feature_selection as feature_selection
import src.model_components as model_components
import src.load_data as load_data

target_column, feature_columns_macro, feature_columns_tech = load_data.get_features_and_targets()
data = load_data.load()
sequence_length = 50
features_test, features_train, targets_test, targets_train, _, _, _ = feature_engineering.split_data(data.copy(deep = True), feature_columns_macro, target_column, sequence_length)
features_test_tech, features_train_tech, targets_test_tech, targets_train_tech, train_dates, test_dates, scaler = feature_engineering.split_data(data, feature_columns_tech, target_column, sequence_length)
print(features_train_tech.shape)
#model_components.train_model(sequence_length, features_train_tech, features_train, targets_train, save_name)

#macro_model, macro_history = model_components.create_model(sequence_length, feature_columns_macro, features_train, targets_train,"macro_model")
#predictions = macro_model.predict(features_test)
#technical_model, technical_history = model_components.create_technical_model(sequence_length, feature_columns_tech, features_train_tech, targets_train_tech)
#technical_predictions = technical_model.predict(features_test_tech)
#evaluate.directional_accuracy(technical_predictions, targets_test_tech)

#predictions = target_scaler.inverse_transform(predictions)
#targets_test = target_scaler.inverse_transform(targets_test)
#training_dates = data["date"][len(data)-len(targets_test):]

#evaluate.backtest(features_test, test_dates)

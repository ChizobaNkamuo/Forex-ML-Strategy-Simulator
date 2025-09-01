import pandas as pd
import numpy as np
import load_data, feature_engineering, model_components, joblib, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def train():
    """
    Train the marcoeconomic and techninical indicator models and save their weights
    """
    sequence_length_tech, sequence_length_macro = 90, 20
    data = load_data.load()
    target_column, feature_columns_macro, feature_columns_tech = load_data.get_features_and_targets()

    features_test_macro, features_train_macro, targets_test_macro, targets_train_macro, test_dates_macro = feature_engineering.split_data(data.copy(deep = True), feature_columns_macro, target_column, sequence_length_macro,"macro_scaler")
    features_test_tech, features_train_tech, targets_test_tech, targets_train_tech, test_dates_tech = feature_engineering.split_data(data, feature_columns_tech, target_column, sequence_length_tech,"tech_scaler")

    model_components.train_macro_model("macro_model", sequence_length_macro, feature_columns_macro, features_train_macro, targets_train_macro )
    model_components.train_tech_model("tech_model", sequence_length_tech, feature_columns_tech, features_train_tech, targets_train_tech)
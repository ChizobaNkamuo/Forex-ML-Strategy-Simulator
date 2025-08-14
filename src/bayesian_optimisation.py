import optuna
import numpy as np
import feature_engineering
import load_data
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, LayerNormalization, Layer, TimeDistributed
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from functools import partial

def objective(trial, data, features, target_column):
    epochs = 75
    n_splits = 5
    
    layers = trial.suggest_int("layers", 1, 4)
    lstm_units = [trial.suggest_categorical(f"lstm_units_{i}", [32, 64, 128]) for i in range(layers)]
    sequence_length = trial.suggest_categorical("sequence_length", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    dropout = trial.suggest_float("dropout", 0, 0.3)
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True),
        ReduceLROnPlateau(patience=5, factor=0.5)
    ]
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(data)):
        features_train, features_val = data[features].iloc[train_idx], data[features].iloc[val_idx]
        targets_train, targets_val = data[target_column].iloc[train_idx], data[target_column].iloc[val_idx]

        scaler = MinMaxScaler()
        features_train, features_val = scaler.fit_transform(features_train), scaler.transform(features_val)
        features_train, targets_train = feature_engineering.create_sequences(features_train, targets_train.values, sequence_length)
        features_val, targets_val = feature_engineering.create_sequences(features_val, targets_val.values, sequence_length)

        model = create_model(trial, len(features))    
        model.fit(features_train, targets_train, epochs=epochs, verbose=0, batch_size = batch_size, shuffle=False, callbacks=callbacks, validation_data=(features_val, targets_val))
   
        prediction = model.predict(features_val)
        score = calculate_profit_accuracy(prediction, targets_val, data["Diff"].iloc[val_idx[sequence_length:]].values)
        scores.append(score)
        
        trial.report(score, fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return np.mean(scores)

def create_model(trial, input_size):
    params = trial.params
    model_layers = []
    for i in range(params["layers"]):
        model_layers.append(LSTM(params[f"lstm_units_{i}"], return_sequences= i != params["layers"] - 1, input_shape=(params["sequence_length"], input_size)))
        model_layers.append(Dropout(params["dropout"]))
    model_layers.append(Dense(3, activation="softmax"))    
    model = Sequential(model_layers)

    model.compile(optimizer=Adam(learning_rate = params["learning_rate"]), loss=SparseCategoricalCrossentropy(), metrics=[])

    return model

def calculate_profit_accuracy(predictions, truth, diff):
    predictions = predictions.squeeze()
    truth = truth.squeeze()
    score = wrong_guesses = 0
    
    for prediction_value, truth_value, diff_value in zip(predictions, truth, diff):
        prediction_value = np.argmax(prediction_value)
        if truth_value != 0: # If the value is increasing or decreasing
            if prediction_value == truth_value:
                score += 1
            elif prediction_value != 0:
                wrong_guesses += 1
        elif prediction_value != 0: # If value is stationary and the prediction is wrong
            if (diff_value > 0 and prediction_value == 2) or (diff_value < 0 and prediction_value == 1):
                score += 1

    denominator = score + wrong_guesses
    return score / denominator if denominator > 0 else 0

data = load_data.load()
target_column, features_macro, features_tech = load_data.get_features_and_targets()

objective_with_data = partial(
    objective,
    data=data,
    features=features_macro,
    target_column=target_column
)

study = optuna.create_study(direction="maximize")
study.optimize(objective_with_data, n_trials=100)
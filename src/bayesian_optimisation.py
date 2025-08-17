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
        features_train, features_val = data.iloc[train_idx].copy(), data.iloc[val_idx].copy()
        features_train, features_val = feature_engineering.add_indicators(features_train)[features], feature_engineering.add_indicators(features_val)[features]
        
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
    features=features_tech,
    target_column=target_column
)

study = optuna.create_study(direction="maximize")
study.optimize(objective_with_data, n_trials=100)

#Trial 28 finished with value: 0.6111600718899811 and parameters: {'layers': 2, 'lstm_units_0': 128, 'lstm_units_1': 32, 'sequence_length': 20, 'learning_rate': 0.004589233717891442, 'batch_size': 64, 'dropout': 0.16353929605134976}.
#Trial 4 finished with value: 0.6278130368366769 and parameters: {'layers': 1, 'lstm_units_0': 64, 'sequence_length': 40, 'learning_rate': 0.006797372520752821, 'batch_size': 64, 'dropout': 0.04419431314881173}.
#Trial 110 finished with value: 0.6388836650699752 and parameters: {'layers': 2, 'lstm_units_0': 128, 'lstm_units_1': 128, 'sequence_length': 90, 'learning_rate': 0.002475868375942953, 'batch_size': 32, 'dropout': 0.18193562077008132}
#Trial 249 finished with value: 0.6714125011930039 and parameters: {'layers': 2, 'lstm_units_0': 128, 'lstm_units_1': 128, 'sequence_length': 30, 'learning_rate': 0.00436878213795238, 'batch_size': 64, 'dropout': 0.1840306801226113}
#Trial 21 finished with value: 0.6805251440625326 and parameters: {'layers': 2, 'lstm_units_0': 64, 'lstm_units_1': 64, 'sequence_length': 100, 'learning_rate': 0.008870914607077226, 'batch_size': 128, 'dropout': 0.14697610473806522}
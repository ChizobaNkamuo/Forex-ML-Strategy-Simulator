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

def objective(trial, data, features, target):
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
    total_trades = trades_made = 0

    for fold, (train_idx, val_idx) in enumerate(tscv.split(data)):
        train_data, _ = feature_engineering.add_indicators(data.iloc[train_idx].copy())
        val_data, _ = feature_engineering.add_indicators(data.iloc[val_idx].copy())
        features_train, features_val = train_data[features], val_data[features]
        targets_train, targets_val = train_data[target], val_data[target]

        scaler = MinMaxScaler()
        features_train, features_val = scaler.fit_transform(features_train), scaler.transform(features_val)
        features_train, targets_train = feature_engineering.create_sequences(features_train, targets_train.values, sequence_length)
        features_val, targets_val = feature_engineering.create_sequences(features_val, targets_val.values, sequence_length)
        _, diff = feature_engineering.create_sequences(val_data[["Diff"]], val_data["Diff"].values, sequence_length)

        model = create_model(trial, len(features))    
        model.fit(features_train, targets_train, epochs=epochs, verbose=0, batch_size = batch_size, shuffle=False, callbacks=callbacks, validation_data=(features_val, targets_val))
        prediction = model.predict(features_val)

        score, num_trades = calculate_profit_accuracy(prediction, targets_val, diff)
        trades_made += num_trades
        total_trades += len(targets_val)

        scores.append(score)
        
        trial.report(score, fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    print(f"Mean number of trades {trades_made/n_splits} out of {total_trades / n_splits} per fold")
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
    return score / denominator if denominator > 0 else 0, denominator

def calculate_sharpe_ratio(predictions, diff):
    pnl = []
    trades_made = 0

    for pred, diff_value in zip(predictions, diff):
        action = np.argmax(pred)
        if action == 1:
            pnl.append(-diff_value)
            trades_made += 1
        elif action == 2:
            pnl.append(diff_value)
            trades_made += 1
        else:
            pnl.append(0)
    pnl = np.array(pnl)
    mean_return = np.mean(pnl)
    std_return  = np.std(pnl)

    return mean_return / std_return if std_return > 0 else 0, trades_made

data = load_data.load()
target, features_macro, features_tech = load_data.get_features_and_targets()

objective_with_data = partial(
    objective,
    data=data,
    features=features_tech,
    target=target
)

study = optuna.create_study(direction="maximize")
study.optimize(objective_with_data, n_trials=100)

#Tech
#Trial 135 finished with value: 0.6395081678622901 and parameters: {'layers': 1, 'lstm_units_0': 32, 'sequence_length': 90, 'learning_rate': 0.0012254914143249621, 'batch_size': 32, 'dropout': 0.04299831983137829}.

#Macro
#Trial 111 finished with value: 0.6294575462362176 and parameters: {'layers': 2, 'lstm_units_0': 128, 'lstm_units_1': 32, 'sequence_length': 20, 'learning_rate': 0.005261567447932985, 'batch_size': 128, 'dropout': 0.10227985946981902}.

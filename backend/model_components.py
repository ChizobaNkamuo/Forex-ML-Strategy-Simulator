import tensorflow as tf
import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def train_macro_model(save_name, sequence_length, features_columns, features_train = None, targets_train = None):
    savePath = os.path.join(MODEL_DIR, f"{save_name}.keras")
    dropout = 0.10227985946981902
    learning_rate = 0.005261567447932985
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(sequence_length, len(features_columns))),
        Dropout(dropout),
        LSTM(32, return_sequences=False, input_shape=(sequence_length, len(features_columns))),
        Dropout(dropout),
        Dense(3, activation="softmax")
    ])
    model.compile(optimizer=Adam(learning_rate = learning_rate), loss=SparseCategoricalCrossentropy(), metrics=[])

    if not features_train:
        model(tf.zeros((1, sequence_length, len(features_columns))))
        model.load_weights(savePath)
        return model
    else:
        reduce_learning_rate = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)
        early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

        checkpoint = ModelCheckpoint(
            savePath,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )

        history = model.fit(features_train, targets_train, 
            epochs=200, 
            batch_size=128, 
            validation_split=0.2,
            verbose=1, 
            shuffle=False,
            callbacks=[reduce_learning_rate, early_stop, checkpoint]
        )
        return model

def train_tech_model(save_name, sequence_length, features_columns, features_train = None, targets_train = None):
    savePath = os.path.join(MODEL_DIR, f"{save_name}.keras")
    dropout = 0.04299831983137829
    learning_rate = 0.0012254914143249621
    model = Sequential([
        LSTM(32, return_sequences=False, input_shape=(sequence_length, len(features_columns))),
        Dropout(dropout),
        Dense(3, activation="softmax")
    ])
    model.compile(optimizer=Adam(learning_rate = learning_rate), loss=SparseCategoricalCrossentropy(), metrics=[])

    if not features_train:
        model(tf.zeros((1, sequence_length, len(features_columns))))
        model.load_weights(savePath)
        return model
    else:
        reduce_learning_rate = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)
        early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

        checkpoint = ModelCheckpoint(
            savePath,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )

        history = model.fit(features_train, targets_train, 
            epochs=200, 
            batch_size=32, 
            validation_split=0.2,
            verbose=1, 
            shuffle=False,
            callbacks=[reduce_learning_rate, early_stop, checkpoint]
        )
        return model
    
def hybrid_predict(macro_predictions, tech_predictions):
    macro_predictions = macro_predictions.squeeze()
    tech_predictions = tech_predictions.squeeze()
    joint_predictions = []
    
    for macro_prediction, tech_prediction in zip(macro_predictions, tech_predictions):
        macro_index = np.argmax(macro_prediction)
        tech_index = np.argmax(tech_prediction)
        if macro_index == tech_index:
            joint_predictions.append(macro_index)
        elif macro_index == 0 or tech_index == 0:
            joint_predictions.append(0) 
        else:
            joint_predictions.append(macro_index if macro_prediction[macro_index] > tech_prediction[tech_index] else tech_index)
    return np.array(joint_predictions)
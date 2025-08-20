import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, LayerNormalization, Layer, TimeDistributed
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras import regularizers
from keras.optimizers import Nadam, Adam
from keras.losses import SparseCategoricalCrossentropy

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],),
                                 initializer="zeros", trainable=True)
        self.u = self.add_weight(name="att_u", shape=(input_shape[-1],),
                                 initializer="glorot_uniform", trainable=True)

        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        v = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        vu = tf.tensordot(v, self.u, axes=1)
        alphas = tf.nn.softmax(vu)

        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), axis=1)
        return output#, alphas
    
def forex_loss(y_true, y_pred, lambd = 0.9, sigma = 0.1):
    #["Open", "Close", "High", "Low"]
    alpha = lambd * (y_true - y_pred)  # Shape: (batch_size, 4)
    beta = sigma * ((y_true[:, 1] + y_true[:, 2])/2 - (y_pred[:, 1] + y_pred[:, 2])/2)  # Difference of high-low averages (wick length difference)
    gamma = sigma * ((y_true[:, 0] + y_true[:, 3])/2 - (y_pred[:, 0] + y_pred[:, 3])/2)  # Difference of open-close averages (body length difference)  
    
    adjusted_errors = tf.stack([# [alpha_i,open - gamma_i, alpha_i,high - beta_i, alpha_i,low - beta_i, alpha_i,close - gamma_i]
        alpha[:, 0] - gamma,
        alpha[:, 1] - beta, 
        alpha[:, 2] - beta,
        alpha[:, 3] - gamma
    ], axis=1)
    
    return tf.reduce_mean(tf.square(adjusted_errors))

def create_candle_stick_model(sequence_length, features_columns, features_train, targets_train):
    model = Sequential([
        LSTM(200, return_sequences=False, input_shape=(sequence_length, len(features_columns)), activation="relu"),
        Dense(4)
    ])

    reduce_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        0.0001,
        decay_steps=1000,
        decay_rate=0.9996,
        staircase=False
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    optimizer = Nadam(
        learning_rate=reduce_learning_rate,#0.001,
        beta_1=0.09,
        beta_2=0.0999,
    )

    checkpoint = ModelCheckpoint(
        "./models/candle_stick_model.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    model.compile(optimizer=optimizer, loss=forex_loss, metrics=[])

    history = model.fit(features_train, targets_train, 
        epochs=200, 
        batch_size=72, 
        validation_split=0.2,
        verbose=1, 
        shuffle=False,
        callbacks=[early_stop, checkpoint]
    )

    return model, history

def mean_abs_directional_loss(y_true, y_pred):
    return tf.reduce_mean(-tf.sign(y_true * y_pred) * tf.math.abs(y_true))

def create_indicator_model(sequence_length, features_columns, features_train, targets_train):
    regularisation = regularizers.l1_l2(l1=1e-6, l2=1e-5)
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, input_shape=(sequence_length, len(features_columns)), kernel_regularizer=regularisation)),
        TimeDistributed(LayerNormalization()),
        Dropout(0.2),
        Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=regularisation)),
        AttentionLayer(),
        Dropout(0.3),
        Dense(25, kernel_regularizer=regularisation),
        Dense(1, kernel_regularizer=regularisation, activation="sigmoid")
    ])

    reduce_learning_rate = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-8)
    early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    optimizer = Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        #epsilon=1e-7, 
        #clipnorm=1.0
    )

    checkpoint = ModelCheckpoint(
        "./models/indicator_model.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    model.compile(optimizer=optimizer, loss="binarycrossentropy", metrics=[])

    history = model.fit(features_train, targets_train, 
        epochs=200, 
        batch_size=64, 
        validation_split=0.2,
        verbose=1, 
        shuffle=False,
        callbacks=[reduce_learning_rate, early_stop, checkpoint]
    )

    return model, history

def train_model(sequence_length, features_columns, features_train, targets_train, save_name):
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(sequence_length, len(features_columns))),
        Dense(3, activation="softmax")
    ])

    reduce_learning_rate = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-8)
    early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

    checkpoint = ModelCheckpoint(
        f"./models/{save_name}.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    model.compile(optimizer="adam", loss=SparseCategoricalCrossentropy(), metrics=[])

    history = model.fit(features_train, targets_train, 
        epochs=1, 
        batch_size=64, 
        validation_split=0.2,
        verbose=1, 
        shuffle=False,
        callbacks=[reduce_learning_rate, early_stop, checkpoint]
    )

    return model, history

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
    print(joint_predictions)
    return np.array(joint_predictions)
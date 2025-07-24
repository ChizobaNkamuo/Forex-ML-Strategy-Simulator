import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, LayerNormalization, Layer, TimeDistributed
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.losses import MeanSquaredError, Hinge, Huber
from keras import regularizers
from keras.optimizers import Adam

mse = MeanSquaredError()
hinge = Hinge()

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
    
def forex_loss(y_true, y_pred):
    true_directions = tf.cast(tf.sign(y_true), tf.float32)
    pred_directions = tf.cast(tf.sign(y_pred), tf.float32)

    return 0.3 * hinge(true_directions, pred_directions)  + mse(y_true, y_pred) 

def create(sequence_length, features_columns, features_train, targets_train):
    regularisation = regularizers.l1_l2(l1=1e-6, l2=1e-5)
    model = Sequential([
        Bidirectional(LSTM(32, return_sequences=True, input_shape=(sequence_length, len(features_columns)), kernel_regularizer=regularisation)),
        TimeDistributed(LayerNormalization()),
        Dropout(0.2),
        Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=regularisation)),
        AttentionLayer(),
        Dropout(0.3),
        Dense(25, kernel_regularizer=regularisation),
        Dense(1, kernel_regularizer=regularisation)
    ])

    reduce_learning_rate = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6)
    early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    optimizer = Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7, 
        clipnorm=1.0
    )

    checkpoint = ModelCheckpoint(
        "./models/best_forex_model.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    model.compile(optimizer=optimizer, loss=forex_loss, metrics=[])

    history = model.fit(features_train, targets_train, 
        epochs=100, 
        batch_size=64, 
        validation_split=0.2,
        verbose=1, 
        shuffle=False,
        callbacks=[reduce_learning_rate, early_stop, checkpoint]
    )

    return model, history
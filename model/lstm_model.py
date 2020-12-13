from tensorflow import keras
from tensorflow.keras import layers

def initialize(n_steps, n_features, n_lstm_units_1, alpha_1, alpha_2, n_lstm_units_2, act_func):
    model = keras.Sequential()
    # layer 1
    model.add(layers.LSTM(units = n_lstm_units_1, input_shape = (n_steps, n_features), return_sequences=True))
    model.add(layers.LeakyReLU(alpha=alpha_1))
    # layer 2
    model.add(layers.LSTM(units = n_lstm_units_2))
    model.add(layers.LeakyReLU(alpha=alpha_2))
    # output layer
    model.add(layers.Dense(1, activation=act_func))
    model.summary()
    
    return model
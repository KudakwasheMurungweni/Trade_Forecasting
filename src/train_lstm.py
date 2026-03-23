import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def build_lstm(input_shape, units=64):
    model = Sequential([
        LSTM(units, input_shape=input_shape, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def run_lstm(X_train, y_train, X_test, window=12, units=64, epochs=100, batch_size=16):
    from src.feature_engineering import make_lstm_sequences
    Xs_train, ys_train = make_lstm_sequences(X_train, y_train, window)
    Xs_test,  ys_test  = make_lstm_sequences(X_test,  np.zeros(len(X_test)), window)

    model = build_lstm((window, X_train.shape[1]), units)
    cb = EarlyStopping(patience=15, restore_best_weights=True)
    history = model.fit(
        Xs_train, ys_train,
        validation_split=0.1,
        epochs=epochs, batch_size=batch_size,
        callbacks=[cb], verbose=0,
    )
    preds = model.predict(Xs_test).ravel()
    return model, preds, ys_test, history

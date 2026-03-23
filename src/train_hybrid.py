import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping

def build_hybrid(seq_shape, exog_dim, lstm_units=64):
    # Branch 1 — temporal (LSTM)
    seq_in = Input(shape=seq_shape, name="sequence_input")
    x = LSTM(lstm_units, return_sequences=False)(seq_in)
    x = Dropout(0.2)(x)

    # Branch 2 — structural / macro (dense)
    exog_in = Input(shape=(exog_dim,), name="exog_input")
    e = Dense(32, activation="relu")(exog_in)
    e = Dropout(0.1)(e)

    # Fusion
    merged = Concatenate()([x, e])
    out = Dense(32, activation="relu")(merged)
    out = Dense(1, name="forecast")(out)

    model = Model(inputs=[seq_in, exog_in], outputs=out)
    model.compile(optimizer="adam", loss="mse")
    return model

def run_hybrid(X_train, y_train, X_test, window=12, epochs=100, batch_size=16):
    from src.feature_engineering import make_hybrid_inputs
    seq_tr, exog_tr = make_hybrid_inputs(X_train, X_train, window)
    seq_te, exog_te = make_hybrid_inputs(X_test,  X_test,  window)
    ys_train = y_train[window:]
    ys_test  = np.zeros(len(seq_te))

    model = build_hybrid(
        seq_shape=(window, X_train.shape[1]),
        exog_dim=exog_tr.shape[1],
    )
    cb = EarlyStopping(patience=15, restore_best_weights=True)
    history = model.fit(
        [seq_tr, exog_tr], ys_train,
        validation_split=0.1,
        epochs=epochs, batch_size=batch_size,
        callbacks=[cb], verbose=0,
    )
    preds = model.predict([seq_te, exog_te]).ravel()
    return model, preds, ys_test, history

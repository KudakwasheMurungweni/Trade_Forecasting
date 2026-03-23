import numpy as np

def make_lstm_sequences(X, y, window=12):
    """Reshape flat features into (samples, timesteps, features) for LSTM."""
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def make_hybrid_inputs(X_seq, X_flat, window=12):
    """
    Returns:
      seq_input  — (samples, timesteps, n_features) for LSTM branch
      exog_input — (samples, n_exog) for dense branch (last timestep's macro/structural cols)
    """
    exog_idx = list(range(7, X_flat.shape[1]))   # macro + structural cols (cols 7 onward)
    seq_input, exog_input = [], []
    for i in range(window, len(X_seq)):
        seq_input.append(X_seq[i - window:i])
        exog_input.append(X_flat[i, exog_idx])
    return np.array(seq_input), np.array(exog_input)

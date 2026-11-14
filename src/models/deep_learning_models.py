"""Small TensorFlow sequence models for leakage-aware demand forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd

SUPPORTED_ARCHITECTURES = ("mlp", "gru", "lstm")


def make_sequences(values, lookback: int = 28) -> tuple[np.ndarray, np.ndarray]:
    """Create one-step sequences where each input uses only earlier targets."""
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if lookback < 1 or len(array) <= lookback:
        raise ValueError("values must contain more observations than the positive lookback")
    x = np.stack([array[i - lookback:i] for i in range(lookback, len(array))])
    y = array[lookback:]
    return x[..., np.newaxis], y


def build_sequence_model(architecture: str, lookback: int = 28, seed: int = 42):
    """Build a compact, regularised neural model; import TensorFlow lazily."""
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"architecture must be one of {SUPPORTED_ARCHITECTURES}")
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    regulariser = tf.keras.regularizers.l2(1e-3)
    inputs = tf.keras.Input(shape=(lookback, 1), name="past_customer_counts")
    if architecture == "mlp":
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=regulariser)(x)
        x = tf.keras.layers.Dropout(0.1)(x)
    elif architecture == "gru":
        x = tf.keras.layers.GRU(8, kernel_regularizer=regulariser)(inputs)
        x = tf.keras.layers.Dropout(0.1)(x)
    else:
        x = tf.keras.layers.LSTM(8, kernel_regularizer=regulariser)(inputs)
        x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(1, name="next_day_demand")(x)
    model = tf.keras.Model(inputs, outputs, name=f"{architecture}_demand_forecaster")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss=tf.keras.losses.Huber())
    return model


def fit_sequence_model(
    history,
    architecture: str,
    lookback: int = 28,
    seed: int = 42,
    max_epochs: int = 30,
    patience: int = 4,
    validation_fraction: float = 0.2,
    fixed_epochs: int | None = None,
):
    """Scale from training history only and fit chronological one-step windows."""
    import tensorflow as tf

    values = np.asarray(history, dtype=np.float32).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("training history must be finite after training-only imputation")
    if not 0 <= validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in [0, 0.5)")
    sequence_count = len(values) - lookback
    if sequence_count <= 1:
        raise ValueError("not enough observations to build training sequences")
    split = None
    if fixed_epochs is None:
        split = int(sequence_count * (1.0 - validation_fraction))
        if split <= 0 or split >= sequence_count:
            raise ValueError("not enough sequence windows for a chronological validation tail")
        # Fit the scaler on the chronological fitting segment, not its early-stopping tail.
        scaler_values = values[:lookback + split]
    else:
        scaler_values = values
    centre = float(scaler_values.mean())
    scale = float(scaler_values.std())
    if scale < 1e-8:
        scale = 1.0
    scaled = (values - centre) / scale
    x, y = make_sequences(scaled, lookback)
    model = build_sequence_model(architecture, lookback, seed)

    if fixed_epochs is None:
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, min_delta=1e-4, restore_best_weights=True
        )
        fit = model.fit(
            x[:split], y[:split], validation_data=(x[split:], y[split:]),
            epochs=max_epochs, batch_size=32, shuffle=False, callbacks=[callback], verbose=0,
        )
        best_epoch = int(np.argmin(fit.history["val_loss"]) + 1)
    else:
        if fixed_epochs < 1:
            raise ValueError("fixed_epochs must be positive")
        model.fit(x, y, epochs=fixed_epochs, batch_size=32, shuffle=False, verbose=0)
        best_epoch = int(fixed_epochs)
    return model, centre, scale, best_epoch


def recursive_forecast(
    model,
    history,
    horizon: int,
    lookback: int = 28,
    centre: float = 0.0,
    scale: float = 1.0,
    index: pd.DatetimeIndex | None = None,
):
    """Forecast multiple steps recursively without consuming future observations."""
    if horizon < 1 or scale <= 0:
        raise ValueError("horizon and scale must be positive")
    values = np.asarray(history, dtype=np.float32).reshape(-1)
    if len(values) < lookback or not np.isfinite(values).all():
        raise ValueError("history must be finite and at least as long as lookback")
    scaled_history = ((values - centre) / scale).astype(np.float32).tolist()
    predictions = []
    for _ in range(horizon):
        window = np.asarray(scaled_history[-lookback:], dtype=np.float32).reshape(1, lookback, 1)
        next_scaled = float(np.asarray(model(window, training=False)).reshape(-1)[0])
        scaled_history.append(next_scaled)
        predictions.append(next_scaled * scale + centre)
    if index is not None:
        if len(index) != horizon:
            raise ValueError("future index length must match the forecast horizon")
        return pd.Series(predictions, index=index)
    return np.asarray(predictions, dtype=float)

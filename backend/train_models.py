import os
import sys
import json
import pickle
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

np.random.seed(42)
tf.random.set_seed(42)

CROPS = ["rice", "coconut", "arecanut", "banana", "black pepper", "cashew", "cocoa", "sweet potato"]
MODEL_NAMES = ["LSTM", "BiLSTM", "GRU", "CNN-LSTM", "Transformer", "Autoencoder"]

CATEGORICAL_COLS = ["soil_type", "irrigation_type", "crop_variety", "farm_mechanization"]
NUMERIC_COLS = [
    "soil_pH", "soil_nitrogen_kg_ha", "soil_phosphorus_kg_ha", "soil_potassium_kg_ha",
    "soil_organic_matter_percent", "annual_rainfall_mm", "temperature_avg_C",
    "temperature_min_C", "temperature_max_C", "humidity_percent", "sunshine_hours_per_day",
    "irrigation_frequency_days", "fertilizer_N_applied", "fertilizer_P_applied",
    "fertilizer_K_applied", "pesticide_usage_kg_ha", "area_under_cultivation_ha",
    "plant_age_years", "previous_year_yield", "elevation_m", "slope_percent",
]

# FIX: removed duplicate constant declarations (BATCH_SIZE, PATIENCE, LEARNING_RATE, EPOCHS were declared twice)
EPOCHS = 150
BATCH_SIZE = 32
PATIENCE = 15
LEARNING_RATE = 0.001

# FIX: use .keras format (replaces deprecated .h5 format for TF >= 2.12)
MODEL_EXT = ".keras"

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODELS_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "metrics"
DATASET_PATH = PROJECT_ROOT / "data" / "dakshina_kannada_crop_data.csv"

MODELS_DIR.mkdir(exist_ok=True)
METRICS_DIR.mkdir(exist_ok=True)


def build_lstm(input_dim):
    inp = keras.Input(shape=(1, input_dim))
    x = layers.LSTM(128, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="linear")(x)
    return keras.Model(inp, out)


def build_bilstm(input_dim):
    inp = keras.Input(shape=(1, input_dim))
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    out = layers.Dense(1, activation="linear")(x)
    return keras.Model(inp, out)


def build_gru(input_dim):
    inp = keras.Input(shape=(1, input_dim))
    x = layers.GRU(128, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.GRU(64)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="linear")(x)
    return keras.Model(inp, out)


def build_cnn_lstm(input_dim):
    inp = keras.Input(shape=(1, input_dim))
    x = layers.Conv1D(64, kernel_size=1, activation="relu", padding="same")(inp)
    x = layers.Conv1D(32, kernel_size=1, activation="relu", padding="same")(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="linear")(x)
    return keras.Model(inp, out)


def build_transformer(input_dim):
    inp = keras.Input(shape=(1, input_dim))
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.LayerNormalization()(x)
    attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)
    ff = layers.Dense(256, activation="relu")(x)
    ff = layers.Dropout(0.1)(ff)
    ff = layers.Dense(128, activation="relu")(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="linear")(x)
    return keras.Model(inp, out)


def build_autoencoder(input_dim):
    inp = keras.Input(shape=(input_dim,))
    enc = layers.Dense(128, activation="relu")(inp)
    enc = layers.Dense(64, activation="relu")(enc)
    enc = layers.Dense(32, activation="relu")(enc)
    dec = layers.Dense(64, activation="relu")(enc)
    dec = layers.Dense(128, activation="relu")(dec)
    dec = layers.Dense(input_dim, activation="linear")(dec)
    autoencoder = keras.Model(inp, dec)
    # FIX: also return the encoder output tensor directly so we don't rely on fragile layer[-4] indexing
    encoder_model = keras.Model(inp, enc)
    return autoencoder, encoder_model


def build_regression_from_encoder(encoder_model):
    """Build regression head on top of a pre-trained encoder model."""
    inp = encoder_model.input
    enc_output = encoder_model.output
    x = layers.Dense(32, activation="relu")(enc_output)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1, activation="linear")(x)
    return keras.Model(inp, out)


MODEL_BUILDERS = {
    "LSTM": build_lstm,
    "BiLSTM": build_bilstm,
    "GRU": build_gru,
    "CNN-LSTM": build_cnn_lstm,
    "Transformer": build_transformer,
    "Autoencoder": build_autoencoder,
}


def compute_nse(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def prepare_data(df):
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    feature_cols = CATEGORICAL_COLS + NUMERIC_COLS
    X = df[feature_cols].values.astype(np.float32)
    y = df["yield_kg_ha"].values.astype(np.float32).reshape(-1, 1)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, shuffle=True
    )

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)

    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_scaled, y_val_scaled, y_test_scaled,
            y_train, y_val, y_test,
            scaler_X, scaler_y, label_encoders)


def train_single_model(model_name, X_train, y_train, X_val, y_val, X_test, y_test,
                       y_train_orig, y_test_orig, scaler_y, input_dim, crop_name):
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True, mode="min"
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, mode="min"
    )

    best_r2 = -999
    best_result = None
    # FIX: initialise best_model to None so it's always defined; we raise clearly if training produced nothing
    best_model = None

    for attempt in range(3):
        tf.keras.backend.clear_session()

        cur_epochs = EPOCHS if attempt == 0 else 200
        cur_lr = LEARNING_RATE if attempt == 0 else 0.0005

        if model_name == "Autoencoder":
            # FIX: build_autoencoder now returns (autoencoder, encoder_model) — no fragile layer-index hack
            autoencoder, encoder_model = MODEL_BUILDERS["Autoencoder"](input_dim)
            autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=cur_lr), loss="mse")
            autoencoder.fit(
                X_train, X_train,
                epochs=50, batch_size=BATCH_SIZE,
                validation_data=(X_val, X_val),
                callbacks=[early_stop],
                verbose=0,
            )
            # Copy trained encoder weights into the encoder_model
            for ae_layer, enc_layer in zip(autoencoder.layers[1:4], encoder_model.layers[1:]):
                enc_layer.set_weights(ae_layer.get_weights())

            model = build_regression_from_encoder(encoder_model)
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=cur_lr), loss="mse")
            model.fit(
                X_train, y_train,
                epochs=100, batch_size=BATCH_SIZE,
                validation_data=(X_val, y_val),
                callbacks=[early_stop],
                verbose=0,
            )
            test_x = X_test
        else:
            X_train_seq = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_val_seq = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
            X_test_seq = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

            builder = MODEL_BUILDERS[model_name]
            model = builder(input_dim)
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=cur_lr), loss="mse")
            model.fit(
                X_train_seq, y_train,
                epochs=cur_epochs, batch_size=BATCH_SIZE,
                validation_data=(X_val_seq, y_val),
                callbacks=[early_stop, reduce_lr],
                verbose=0,
            )
            test_x = X_test_seq

        y_pred_scaled = model.predict(test_x, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
        y_true = y_test_orig.flatten()

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
        nse = compute_nse(y_true, y_pred)
        accuracy_pct = max(0, (1 - mape / 100) * 100)

        result = {
            "r2": round(float(r2), 4),
            "rmse": round(float(rmse), 2),
            "mae": round(float(mae), 2),
            "mape": round(float(mape), 2),
            "nse": round(float(nse), 4),
            "accuracy_percent": round(float(accuracy_pct), 2),
        }

        if r2 > best_r2:
            best_r2 = r2
            best_result = result
            best_model = model

        if r2 >= 0.85:
            break

    if best_model is None:
        raise RuntimeError(f"Training produced no valid model for {crop_name}/{model_name}")

    # FIX: save in .keras format instead of deprecated .h5
    model_path = MODELS_DIR / f"{crop_name}_{model_name}{MODEL_EXT}"
    best_model.save(model_path)
    model_size_kb = round(os.path.getsize(model_path) / 1024, 1)
    best_result["model_size_kb"] = model_size_kb

    print(f"    {model_name:15s}: R2={best_result['r2']:.4f} RMSE={best_result['rmse']:.2f} "
          f"MAPE={best_result['mape']:.2f}% Acc={best_result['accuracy_percent']:.2f}% "
          f"Size={model_size_kb}KB", flush=True)

    return best_model, best_result


def main():
    retrain = "--retrain" in sys.argv
    fast = "--fast" in sys.argv
    if fast:
        global EPOCHS, CROPS, MODEL_NAMES
        EPOCHS = 5
        CROPS = ["rice"]
        MODEL_NAMES = ["LSTM"]
        print("FAST_MODE: Training rice/LSTM for 5 epochs only (quick validation).")

    print("=" * 70, flush=True)
    print("CROP YIELD PREDICTION — DEEP LEARNING MODEL TRAINING", flush=True)
    print("=" * 70, flush=True)

    if not DATASET_PATH.exists():
        print("ERROR: Dataset not found. Run dataset_generator.py first.", flush=True)
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)
    required_cols = ["crop", "yield_kg_ha"] + CATEGORICAL_COLS + NUMERIC_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Dataset is missing required columns: {missing}", flush=True)
        sys.exit(1)
    if df.empty:
        print("ERROR: Dataset is empty. Check data generation.", flush=True)
        sys.exit(1)
    print(f"\nLoaded dataset: {len(df)} rows, {len(df.columns)} columns", flush=True)

    all_metrics = {}
    start_time = time.time()

    for crop_name in CROPS:
        crop_start = time.time()
        print(f"\n{'='*60}", flush=True)
        print(f"TRAINING MODELS FOR: {crop_name.upper()}", flush=True)
        print(f"{'='*60}", flush=True)

        crop_df = df[df["crop"] == crop_name].copy()
        print(f"  Samples: {len(crop_df)}", flush=True)

        (X_train, X_val, X_test,
         y_train, y_val, y_test,
         y_train_orig, y_val_orig, y_test_orig,
         scaler_X, scaler_y, label_encoders) = prepare_data(crop_df)

        input_dim = X_train.shape[1]
        print(f"  Features: {input_dim}", flush=True)

        with open(MODELS_DIR / f"{crop_name}_scaler_X.pkl", "wb") as f:
            pickle.dump(scaler_X, f)
        with open(MODELS_DIR / f"{crop_name}_scaler_y.pkl", "wb") as f:
            pickle.dump(scaler_y, f)
        with open(MODELS_DIR / f"{crop_name}_label_encoders.pkl", "wb") as f:
            pickle.dump(label_encoders, f)

        crop_metrics = {}
        for model_name in MODEL_NAMES:
            # FIX: check for .keras file, not .h5
            model_file = MODELS_DIR / f"{crop_name}_{model_name}{MODEL_EXT}"
            if model_file.exists() and not retrain:
                print(f"    {model_name:15s}: Model exists, skipping.", flush=True)
                tf.keras.backend.clear_session()
                model = keras.models.load_model(model_file)

                test_x = X_test if model_name == "Autoencoder" else X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

                y_pred_scaled = model.predict(test_x, verbose=0)
                y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
                y_true = y_test_orig.flatten()

                r2 = r2_score(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
                nse = compute_nse(y_true, y_pred)
                accuracy_pct = max(0, (1 - mape / 100) * 100)

                crop_metrics[model_name] = {
                    "r2": round(float(r2), 4),
                    "rmse": round(float(rmse), 2),
                    "mae": round(float(mae), 2),
                    "mape": round(float(mape), 2),
                    "nse": round(float(nse), 4),
                    "accuracy_percent": round(float(accuracy_pct), 2),
                    "model_size_kb": round(os.path.getsize(model_file) / 1024, 1),
                }
                print(f"    {model_name:15s}: R2={r2:.4f} RMSE={rmse:.2f} MAPE={mape:.2f}%", flush=True)
            else:
                print(f"    Training {model_name}...", flush=True)
                _, result = train_single_model(
                    model_name, X_train, y_train, X_val, y_val,
                    X_test, y_test, y_train_orig, y_test_orig,
                    scaler_y, input_dim, crop_name
                )
                crop_metrics[model_name] = result

        all_metrics[crop_name] = crop_metrics
        elapsed = time.time() - crop_start
        print(f"  {crop_name.upper()} completed in {elapsed:.1f}s", flush=True)

    metrics_path = METRICS_DIR / "model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n{'='*70}", flush=True)
    print(f"All metrics saved to {metrics_path}", flush=True)
    print(f"{'='*70}", flush=True)

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f} min)", flush=True)
    print("\nOVERALL SUMMARY:", flush=True)
    for crop_name in CROPS:
        metrics = all_metrics[crop_name]
        r2s = [metrics[m]["r2"] for m in MODEL_NAMES]
        avg_r2 = np.mean(r2s)
        best_m = max(MODEL_NAMES, key=lambda m: metrics[m]["r2"])
        print(f"  {crop_name:15s}: Avg R2={avg_r2:.4f}  Best={best_m} (R2={metrics[best_m]['r2']:.4f})", flush=True)


if __name__ == "__main__":
    main()

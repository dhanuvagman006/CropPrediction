import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

CATEGORICAL_COLS = ["soil_type", "irrigation_type", "crop_variety", "farm_mechanization"]
NUMERIC_COLS = [
    "soil_pH", "soil_nitrogen_kg_ha", "soil_phosphorus_kg_ha", "soil_potassium_kg_ha",
    "soil_organic_matter_percent", "annual_rainfall_mm", "temperature_avg_C",
    "temperature_min_C", "temperature_max_C", "humidity_percent", "sunshine_hours_per_day",
    "irrigation_frequency_days", "fertilizer_N_applied", "fertilizer_P_applied",
    "fertilizer_K_applied", "pesticide_usage_kg_ha", "area_under_cultivation_ha",
    "plant_age_years", "previous_year_yield", "elevation_m", "slope_percent",
]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

CROP_YIELD_RANGES = {
    "rice": (2800, 4500),
    "coconut": (9000, 14000),
    "arecanut": (1800, 3200),
    "banana": (25000, 40000),
    "black pepper": (800, 2200),
    "cashew": (600, 1800),
    "cocoa": (700, 1500),
    "sweet potato": (8000, 18000),
}

_loaded_models = {}
_loaded_scalers = {}
_loaded_encoders = {}


def _load_model(crop, model_name):
    key = f"{crop}_{model_name}"
    if key not in _loaded_models:
        path = f"models/{crop}_{model_name}.h5"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        _loaded_models[key] = keras.models.load_model(path)
    return _loaded_models[key]


def _load_scalers(crop):
    if crop not in _loaded_scalers:
        with open(f"models/{crop}_scaler_X.pkl", "rb") as f:
            scaler_x = pickle.load(f)
        with open(f"models/{crop}_scaler_y.pkl", "rb") as f:
            scaler_y = pickle.load(f)
        _loaded_scalers[crop] = (scaler_x, scaler_y)
    return _loaded_scalers[crop]


def _load_encoders(crop):
    if crop not in _loaded_encoders:
        with open(f"models/{crop}_label_encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        _loaded_encoders[crop] = encoders
    return _loaded_encoders[crop]


def encode_categoricals(features, crop):
    encoders = _load_encoders(crop)
    encoded = {}
    for col in CATEGORICAL_COLS:
        val = features.get(col)
        if val is None:
            raise ValueError(f"Missing categorical feature: {col}")
        le = encoders[col]
        val_str = str(val)
        known = list(le.classes_)
        if val_str not in known:
            mapped = {
                "soil_type": {"laterite": "Laterite", "red loam": "Red Loam", "clay loam": "Clay Loam", "sandy loam": "Sandy Loam"},
                "irrigation_type": {"rainfed": "Rainfed", "drip": "Drip", "sprinkler": "Sprinkler", "canal": "Canal"},
            }
            if col in mapped:
                val_str = mapped[col].get(val_str.lower(), known[0])
            elif col == "crop_variety":
                val_str = "1" if str(val) in ("1", "True", "HYV", "hyv") else "0"
            elif col == "farm_mechanization":
                val_str = "1" if str(val) in ("1", "True", "yes") else "0"
        try:
            encoded[col] = le.transform([val_str])[0]
        except ValueError:
            encoded[col] = 0
    return encoded


def predict_yield(crop, model_name, features):
    crop = crop.lower().strip()
    model_name = model_name.strip()

    scaler_x, scaler_y = _load_scalers(crop)
    encoders = _load_encoders(crop)

    encoded_cats = encode_categoricals(features, crop)

    feature_vector = []
    for col in FEATURE_COLS:
        if col in CATEGORICAL_COLS:
            feature_vector.append(float(encoded_cats[col]))
        else:
            val = features.get(col)
            if val is None:
                raise ValueError(f"Missing feature: {col}")
            feature_vector.append(float(val))

    X = np.array(feature_vector).reshape(1, -1).astype(np.float32)
    X_scaled = scaler_x.transform(X)

    model = _load_model(crop, model_name)

    if model_name in ("LSTM", "BiLSTM", "GRU", "CNN-LSTM", "Transformer"):
        X_input = X_scaled.reshape(1, 1, X_scaled.shape[1])
    else:
        X_input = X_scaled

    y_pred_scaled = model.predict(X_input, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0][0]

    y_min, y_max = CROP_YIELD_RANGES.get(crop, (0, 10000))
    y_pred = max(y_min, min(y_max, y_pred))

    ci_lower = y_pred * 0.90
    ci_upper = y_pred * 1.10

    yield_range = y_max - y_min
    ratio = (y_pred - y_min) / yield_range
    if ratio < 0.25:
        category = "Poor"
    elif ratio < 0.50:
        category = "Average"
    elif ratio < 0.75:
        category = "Good"
    else:
        category = "Excellent"

    return {
        "predicted_yield": round(float(y_pred), 2),
        "unit": "kg/ha",
        "crop": crop,
        "model": model_name,
        "confidence_interval": {
            "lower": round(float(ci_lower), 2),
            "upper": round(float(ci_upper), 2),
        },
        "yield_category": category,
    }


def unload_all():
    global _loaded_models, _loaded_scalers, _loaded_encoders
    _loaded_models.clear()
    _loaded_scalers.clear()
    _loaded_encoders.clear()


if __name__ == "__main__":
    test_features = {
        "soil_type": "Laterite",
        "soil_pH": 5.5,
        "soil_nitrogen_kg_ha": 220,
        "soil_phosphorus_kg_ha": 35,
        "soil_potassium_kg_ha": 180,
        "soil_organic_matter_percent": 2.0,
        "annual_rainfall_mm": 3500,
        "temperature_avg_C": 27,
        "temperature_min_C": 21,
        "temperature_max_C": 32,
        "humidity_percent": 80,
        "sunshine_hours_per_day": 5.5,
        "irrigation_type": "Canal",
        "irrigation_frequency_days": 5,
        "fertilizer_N_applied": 120,
        "fertilizer_P_applied": 45,
        "fertilizer_K_applied": 80,
        "pesticide_usage_kg_ha": 2.0,
        "area_under_cultivation_ha": 3.0,
        "farm_mechanization": 1,
        "crop_variety": 1,
        "plant_age_years": 0,
        "previous_year_yield": 3500,
        "elevation_m": 50,
        "slope_percent": 5,
    }

    result = predict_yield("rice", "LSTM", test_features)
    print(json.dumps(result, indent=2))

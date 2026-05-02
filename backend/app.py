import os
import sys
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict_yield, CROP_YIELD_RANGES, FEATURE_COLS, CATEGORICAL_COLS, NUMERIC_COLS

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

CROPS = ["rice", "coconut", "arecanut", "banana", "black pepper", "cashew", "cocoa", "sweet potato"]
MODEL_NAMES = ["LSTM", "BiLSTM", "GRU", "CNN-LSTM", "Transformer", "Autoencoder"]

CROP_METADATA = {
    "rice": {
        "name": "Rice",
        "season": "Kharif (Jun–Nov)",
        "soil_type": "Clay Loam, Laterite",
        "yield_min_kg_ha": 2800,
        "yield_max_kg_ha": 4500,
        "optimal_rainfall_mm": "3000–4000",
        "optimal_temp_C": "25–30",
        "description": "Staple food crop grown extensively in paddy fields across the coastal plains and midland regions. Paddy cultivation is integral to the agricultural economy and food security of Dakshina Kannada.",
        "districts": "Puttur, Bantwal, Belthangady",
        "icon": "🌾",
    },
    "coconut": {
        "name": "Coconut",
        "season": "Perennial (Year-round)",
        "soil_type": "Laterite, Sandy Loam",
        "yield_min_kg_ha": 9000,
        "yield_max_kg_ha": 14000,
        "optimal_rainfall_mm": "2800–4000",
        "optimal_temp_C": "26–30",
        "description": "Coconut palms are ubiquitous across Dakshina Kannada, providing copra, tender coconut water, and raw materials for coir industry. A vital cash crop supporting rural livelihoods.",
        "districts": "Mangaluru, Ullal, Mulki",
        "icon": "🥥",
    },
    "arecanut": {
        "name": "Arecanut",
        "season": "Perennial (Year-round)",
        "soil_type": "Laterite, Red Loam",
        "yield_min_kg_ha": 1800,
        "yield_max_kg_ha": 3200,
        "optimal_rainfall_mm": "3000–4200",
        "optimal_temp_C": "25–30",
        "description": "Arecanut (betel nut) is the most important cash crop of Dakshina Kannada, contributing significantly to the state's arecanut production. Grown in homestead gardens and dedicated plantations.",
        "districts": "Sullia, Puttur, Beltangady",
        "icon": "🌴",
    },
    "banana": {
        "name": "Banana",
        "season": "Annual (Year-round planting)",
        "soil_type": "Red Loam, Clay Loam",
        "yield_min_kg_ha": 25000,
        "yield_max_kg_ha": 40000,
        "optimal_rainfall_mm": "2800–3800",
        "optimal_temp_C": "26–30",
        "description": "Banana cultivation is widespread in the fertile plains and homestead gardens. High-yielding varieties with drip irrigation achieve the best results in the tropical climate of the region.",
        "districts": "Mangaluru, Bantwal, Moodbidri",
        "icon": "🍌",
    },
    "black pepper": {
        "name": "Black Pepper",
        "season": "Perennial (Year-round)",
        "soil_type": "Laterite (hilly tracts)",
        "yield_min_kg_ha": 800,
        "yield_max_kg_ha": 2200,
        "optimal_rainfall_mm": "3200–4500",
        "optimal_temp_C": "23–28",
        "description": "Known as the 'King of Spices', black pepper is cultivated on the hilly slopes of the Western Ghats foothills in Dakshina Kannada. Often grown as an intercrop with coconut and arecanut.",
        "districts": "Sullia, Puttur, Subramanya",
        "icon": "🌶️",
    },
    "cashew": {
        "name": "Cashew",
        "season": "Perennial (Year-round)",
        "soil_type": "Laterite (coastal laterite)",
        "yield_min_kg_ha": 600,
        "yield_max_kg_ha": 1800,
        "optimal_rainfall_mm": "2800–3600",
        "optimal_temp_C": "25–30",
        "description": "Cashew is well-suited to the lateritic soils of coastal Dakshina Kannada. The region has extensive cashew plantations on undulating terrain, producing high-quality raw nuts for processing.",
        "districts": "Ullal, Mangaluru, Mulki",
        "icon": "🥜",
    },
    "cocoa": {
        "name": "Cocoa",
        "season": "Perennial (Year-round)",
        "soil_type": "Laterite, Red Loam",
        "yield_min_kg_ha": 700,
        "yield_max_kg_ha": 1500,
        "optimal_rainfall_mm": "3000–4200",
        "optimal_temp_C": "24–28",
        "description": "Cocoa is increasingly promoted as an intercrop in coconut and arecanut gardens, providing additional income. The humid tropical climate of Dakshina Kannada is well-suited for cocoa cultivation.",
        "districts": "Puttur, Sullia, Belthangady",
        "icon": "🍫",
    },
    "sweet potato": {
        "name": "Sweet Potato",
        "season": "Rabi (Oct–Feb)",
        "soil_type": "Sandy Loam, Red Loam",
        "yield_min_kg_ha": 8000,
        "yield_max_kg_ha": 18000,
        "optimal_rainfall_mm": "2800–3600",
        "optimal_temp_C": "22–28",
        "description": "Sweet potato is a short-duration root crop grown during the Rabi season. It serves as both a food security crop and a source of income, thriving in the well-drained soils of the region.",
        "districts": "Bantwal, Puttur, Moodbidri",
        "icon": "🍠",
    },
}


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        crop = data.get("crop", "").lower().strip()
        model_name = data.get("model", "").strip()
        features = data.get("features", {})

        if not crop or crop not in CROPS:
            return jsonify({"error": f"Invalid crop. Must be one of: {', '.join(CROPS)}"}), 400
        if not model_name or model_name not in MODEL_NAMES:
            return jsonify({"error": f"Invalid model. Must be one of: {', '.join(MODEL_NAMES)}"}), 400
        if not features:
            return jsonify({"error": "Features dictionary is required"}), 400

        result = predict_yield(crop, model_name, features)
        return jsonify(result)

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    try:
        metrics_path = "metrics/model_metrics.json"
        if not os.path.exists(metrics_path):
            return jsonify({"error": "Model metrics not found. Run train_models.py first."}), 404
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": f"Failed to load metrics: {str(e)}"}), 500


@app.route("/api/crops", methods=["GET"])
def api_crops():
    try:
        crops_list = []
        for crop_key in CROPS:
            meta = CROP_METADATA[crop_key]
            crops_list.append({
                "key": crop_key,
                "name": meta["name"],
                "season": meta["season"],
                "soil_type": meta["soil_type"],
                "yield_min_kg_ha": meta["yield_min_kg_ha"],
                "yield_max_kg_ha": meta["yield_max_kg_ha"],
                "optimal_rainfall_mm": meta["optimal_rainfall_mm"],
                "optimal_temp_C": meta["optimal_temp_C"],
                "description": meta["description"],
                "districts": meta["districts"],
                "icon": meta["icon"],
            })
        return jsonify(crops_list)
    except Exception as e:
        return jsonify({"error": f"Failed to load crop data: {str(e)}"}), 500


@app.route("/api/feature-config/<crop>", methods=["GET"])
def api_feature_config(crop):
    try:
        crop = crop.lower().strip()
        if crop not in CROPS:
            return jsonify({"error": f"Invalid crop: {crop}"}), 400

        config = {
            "crop": crop,
            "features": {},
        }

        feature_info = {
            "soil_pH": {
                "type": "numeric", "label": "Soil pH",
                "min": 4.5, "max": 6.5, "step": 0.1, "default": 5.5,
                "group": "soil", "tooltip": "Soil acidity level. DK soils are typically acidic (4.5–6.5)."
            },
            "soil_nitrogen_kg_ha": {
                "type": "numeric", "label": "Soil Nitrogen (kg/ha)",
                "min": 100, "max": 350, "step": 1, "default": 220,
                "group": "soil", "tooltip": "Available nitrogen content in the soil."
            },
            "soil_phosphorus_kg_ha": {
                "type": "numeric", "label": "Soil Phosphorus (kg/ha)",
                "min": 10, "max": 60, "step": 1, "default": 35,
                "group": "soil", "tooltip": "Available phosphorus content in the soil."
            },
            "soil_potassium_kg_ha": {
                "type": "numeric", "label": "Soil Potassium (kg/ha)",
                "min": 80, "max": 300, "step": 1, "default": 180,
                "group": "soil", "tooltip": "Available potassium content in the soil."
            },
            "soil_organic_matter_percent": {
                "type": "numeric", "label": "Soil Organic Matter (%)",
                "min": 0.8, "max": 3.5, "step": 0.1, "default": 2.0,
                "group": "soil", "tooltip": "Percentage of organic matter in the soil."
            },
            "soil_type": {
                "type": "categorical", "label": "Soil Type",
                "options": ["Laterite", "Red Loam", "Clay Loam", "Sandy Loam"],
                "default": "Laterite",
                "group": "soil", "tooltip": "Dominant soil type. Laterite is most common in DK."
            },
            "annual_rainfall_mm": {
                "type": "numeric", "label": "Annual Rainfall (mm)",
                "min": 2800, "max": 4500, "step": 10, "default": 3500,
                "group": "climate", "tooltip": "Total annual rainfall. DK receives 2800–4500 mm."
            },
            "temperature_avg_C": {
                "type": "numeric", "label": "Avg Temperature (°C)",
                "min": 22, "max": 32, "step": 0.5, "default": 27,
                "group": "climate", "tooltip": "Average annual temperature."
            },
            "temperature_min_C": {
                "type": "numeric", "label": "Min Temperature (°C)",
                "min": 18, "max": 24, "step": 0.5, "default": 21,
                "group": "climate", "tooltip": "Minimum temperature during the growing season."
            },
            "temperature_max_C": {
                "type": "numeric", "label": "Max Temperature (°C)",
                "min": 28, "max": 36, "step": 0.5, "default": 32,
                "group": "climate", "tooltip": "Maximum temperature during the growing season."
            },
            "humidity_percent": {
                "type": "numeric", "label": "Humidity (%)",
                "min": 65, "max": 92, "step": 1, "default": 80,
                "group": "climate", "tooltip": "Average relative humidity. DK is very humid."
            },
            "sunshine_hours_per_day": {
                "type": "numeric", "label": "Sunshine Hours/Day",
                "min": 3.5, "max": 7.5, "step": 0.5, "default": 5.5,
                "group": "climate", "tooltip": "Average daily sunshine hours."
            },
            "irrigation_type": {
                "type": "categorical", "label": "Irrigation Type",
                "options": ["Rainfed", "Drip", "Sprinkler", "Canal"],
                "default": "Rainfed",
                "group": "irrigation", "tooltip": "Primary irrigation method used."
            },
            "irrigation_frequency_days": {
                "type": "numeric", "label": "Irrigation Frequency (days)",
                "min": 0, "max": 7, "step": 1, "default": 0,
                "group": "irrigation", "tooltip": "Days between irrigation cycles. 0 = rainfed."
            },
            "farm_mechanization": {
                "type": "categorical", "label": "Farm Mechanization",
                "options": ["Manual (0)", "Mechanized (1)"],
                "default": "Mechanized (1)",
                "group": "irrigation", "tooltip": "Level of farm mechanization."
            },
            "fertilizer_N_applied": {
                "type": "numeric", "label": "Fertilizer N Applied (kg/ha)",
                "min": 60, "max": 200, "step": 1, "default": 120,
                "group": "irrigation", "tooltip": "Nitrogen fertilizer applied per hectare."
            },
            "fertilizer_P_applied": {
                "type": "numeric", "label": "Fertilizer P Applied (kg/ha)",
                "min": 20, "max": 80, "step": 1, "default": 45,
                "group": "irrigation", "tooltip": "Phosphorus fertilizer applied per hectare."
            },
            "fertilizer_K_applied": {
                "type": "numeric", "label": "Fertilizer K Applied (kg/ha)",
                "min": 40, "max": 150, "step": 1, "default": 80,
                "group": "irrigation", "tooltip": "Potassium fertilizer applied per hectare."
            },
            "pesticide_usage_kg_ha": {
                "type": "numeric", "label": "Pesticide Usage (kg/ha)",
                "min": 0.5, "max": 4.5, "step": 0.1, "default": 2.0,
                "group": "irrigation", "tooltip": "Total pesticides applied per hectare."
            },
            "crop_variety": {
                "type": "categorical", "label": "Crop Variety",
                "options": ["Traditional (0)", "HYV (1)"],
                "default": "HYV (1)",
                "group": "crop", "tooltip": "High Yielding Variety (HYV) or Traditional variety."
            },
            "plant_age_years": {
                "type": "numeric", "label": "Plant Age (years)",
                "min": 0, "max": 35, "step": 1, "default": 0,
                "group": "crop", "tooltip": "Age of perennial plants. Set to 0 for seasonal crops."
            },
            "area_under_cultivation_ha": {
                "type": "numeric", "label": "Cultivation Area (ha)",
                "min": 0.5, "max": 10, "step": 0.5, "default": 3.0,
                "group": "crop", "tooltip": "Total area under cultivation in hectares."
            },
            "previous_year_yield": {
                "type": "numeric", "label": "Previous Year Yield (kg/ha)",
                "min": 500, "max": 40000, "step": 100, "default": 3500,
                "group": "crop", "tooltip": "Yield obtained in the previous year."
            },
            "elevation_m": {
                "type": "numeric", "label": "Elevation (m)",
                "min": 0, "max": 950, "step": 10, "default": 50,
                "group": "crop", "tooltip": "Elevation above sea level. DK ranges from 0–950 m."
            },
            "slope_percent": {
                "type": "numeric", "label": "Slope (%)",
                "min": 0, "max": 35, "step": 1, "default": 5,
                "group": "crop", "tooltip": "Land slope percentage. Important for erosion and drainage."
            },
        }

        y_min, y_max = CROP_YIELD_RANGES[crop]
        feature_info["previous_year_yield"]["default"] = round((y_min + y_max) / 2, 0)

        if crop in ("coconut", "arecanut", "black pepper", "cashew", "cocoa"):
            feature_info["plant_age_years"]["default"] = 10

        for feat_key, feat_info in feature_info.items():
            config["features"][feat_key] = feat_info

        return jsonify(config)

    except Exception as e:
        return jsonify({"error": f"Failed to load feature config: {str(e)}"}), 500


@app.route("/api/comparison-data", methods=["GET"])
def api_comparison_data():
    try:
        metrics_path = "metrics/model_metrics.json"
        if not os.path.exists(metrics_path):
            return jsonify({"error": "Model metrics not found. Run train_models.py first."}), 404
        with open(metrics_path, "r") as f:
            all_metrics = json.load(f)

        models = MODEL_NAMES
        crops = CROPS

        metrics_table = []
        for model in models:
            row = {"model": model}
            total_r2 = 0
            total_rmse = 0
            total_mae = 0
            total_mape = 0
            total_nse = 0
            total_acc = 0
            total_size = 0
            count = 0
            for crop in crops:
                if crop in all_metrics and model in all_metrics[crop]:
                    m = all_metrics[crop][model]
                    row[f"{crop}_r2"] = m["r2"]
                    row[f"{crop}_rmse"] = m["rmse"]
                    row[f"{crop}_mae"] = m["mae"]
                    row[f"{crop}_mape"] = m["mape"]
                    row[f"{crop}_nse"] = m["nse"]
                    row[f"{crop}_accuracy_percent"] = m["accuracy_percent"]
                    row[f"{crop}_model_size_kb"] = m["model_size_kb"]
                    total_r2 += m["r2"]
                    total_rmse += m["rmse"]
                    total_mae += m["mae"]
                    total_mape += m["mape"]
                    total_nse += m["nse"]
                    total_acc += m["accuracy_percent"]
                    total_size += m["model_size_kb"]
                    count += 1
            if count > 0:
                row["avg_r2"] = round(total_r2 / count, 4)
                row["avg_rmse"] = round(total_rmse / count, 2)
                row["avg_mae"] = round(total_mae / count, 2)
                row["avg_mape"] = round(total_mape / count, 2)
                row["avg_nse"] = round(total_nse / count, 4)
                row["avg_accuracy"] = round(total_acc / count, 2)
                row["avg_size"] = round(total_size / count, 1)
            metrics_table.append(row)

        radar_data = {}
        for model in models:
            avg_r2 = 0
            avg_nse = 0
            avg_acc = 0
            avg_mape_inv = 0
            avg_rmse_norm = 0
            count = 0
            for crop in crops:
                if crop in all_metrics and model in all_metrics[crop]:
                    m = all_metrics[crop][model]
                    avg_r2 += m["r2"]
                    avg_nse += m["nse"]
                    avg_acc += m["accuracy_percent"]
                    avg_mape_inv += max(0, 100 - m["mape"])
                    avg_rmse_norm += m["rmse"]
                    count += 1
            if count > 0:
                max_rmse = max(all_metrics[c][model]["rmse"] for c in crops if c in all_metrics and model in all_metrics[c]) if count > 0 else 1
                radar_data[model] = {
                    "r2": round(avg_r2 / count, 4),
                    "nse": round(avg_nse / count, 4),
                    "accuracy": round(avg_acc / count, 2),
                    "mape_inv": round(avg_mape_inv / count, 2),
                    "rmse_norm": round((avg_rmse_norm / count) / max(max_rmse, 1), 4),
                }

        bar_data = {}
        for metric in ["r2", "accuracy_percent", "mape", "nse"]:
            bar_data[metric] = {}
            for model in models:
                bar_data[metric][model] = []
                for crop in crops:
                    if crop in all_metrics and model in all_metrics[crop]:
                        bar_data[metric][model].append(all_metrics[crop][model][metric])
                    else:
                        bar_data[metric][model].append(0)

        heatmap_data = []
        for crop in crops:
            row = {"crop": crop}
            for model in models:
                if crop in all_metrics and model in all_metrics[crop]:
                    row[model] = all_metrics[crop][model]["accuracy_percent"]
                else:
                    row[model] = 0
            heatmap_data.append(row)

        return jsonify({
            "models": models,
            "crops": crops,
            "metrics_table": metrics_table,
            "radar_data": radar_data,
            "bar_data": bar_data,
            "heatmap_data": heatmap_data,
        })

    except Exception as e:
        return jsonify({"error": f"Failed to load comparison data: {str(e)}"}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("AgriPredict DK — Crop Yield Prediction API")
    print("Running on http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)

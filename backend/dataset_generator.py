import numpy as np
import pandas as pd
import os

np.random.seed(42)

ROWS_PER_CROP = 2200

CROPS = {
    "rice": {
        "season": "Kharif",
        "yield_min": 2800,
        "yield_max": 4500,
        "yield_mid": 3650,
        "yield_half_range": 850,
        "optimal_ph": 5.5,
        "optimal_temp": 27.5,
        "rainfall_range": (3000, 4200),
        "soil_weights": [0.40, 0.30, 0.25, 0.05],
        "irrigation_weights": [0.15, 0.30, 0.15, 0.40],
        "perennial": False,
    },
    "coconut": {
        "season": "Annual",
        "yield_min": 9000,
        "yield_max": 14000,
        "yield_mid": 11500,
        "yield_half_range": 2500,
        "optimal_ph": 5.5,
        "optimal_temp": 28.0,
        "rainfall_range": (2800, 4000),
        "soil_weights": [0.55, 0.25, 0.10, 0.10],
        "irrigation_weights": [0.20, 0.45, 0.25, 0.10],
        "perennial": True,
    },
    "arecanut": {
        "season": "Annual",
        "yield_min": 1800,
        "yield_max": 3200,
        "yield_mid": 2500,
        "yield_half_range": 700,
        "optimal_ph": 5.5,
        "optimal_temp": 27.0,
        "rainfall_range": (3000, 4200),
        "soil_weights": [0.60, 0.20, 0.12, 0.08],
        "irrigation_weights": [0.15, 0.40, 0.25, 0.20],
        "perennial": True,
    },
    "banana": {
        "season": "Annual",
        "yield_min": 25000,
        "yield_max": 40000,
        "yield_mid": 32500,
        "yield_half_range": 7500,
        "optimal_ph": 5.8,
        "optimal_temp": 28.0,
        "rainfall_range": (2800, 3800),
        "soil_weights": [0.30, 0.35, 0.25, 0.10],
        "irrigation_weights": [0.05, 0.50, 0.20, 0.25],
        "perennial": False,
    },
    "black pepper": {
        "season": "Annual",
        "yield_min": 800,
        "yield_max": 2200,
        "yield_mid": 1500,
        "yield_half_range": 700,
        "optimal_ph": 5.5,
        "optimal_temp": 26.0,
        "rainfall_range": (3200, 4500),
        "soil_weights": [0.70, 0.15, 0.10, 0.05],
        "irrigation_weights": [0.60, 0.20, 0.10, 0.10],
        "perennial": True,
    },
    "cashew": {
        "season": "Annual",
        "yield_min": 600,
        "yield_max": 1800,
        "yield_mid": 1200,
        "yield_half_range": 600,
        "optimal_ph": 5.2,
        "optimal_temp": 27.0,
        "rainfall_range": (2800, 3600),
        "soil_weights": [0.75, 0.10, 0.05, 0.10],
        "irrigation_weights": [0.50, 0.25, 0.10, 0.15],
        "perennial": True,
    },
    "cocoa": {
        "season": "Annual",
        "yield_min": 700,
        "yield_max": 1500,
        "yield_mid": 1100,
        "yield_half_range": 400,
        "optimal_ph": 5.8,
        "optimal_temp": 26.5,
        "rainfall_range": (3000, 4200),
        "soil_weights": [0.55, 0.25, 0.12, 0.08],
        "irrigation_weights": [0.55, 0.25, 0.10, 0.10],
        "perennial": True,
    },
    "sweet potato": {
        "season": "Rabi",
        "yield_min": 8000,
        "yield_max": 18000,
        "yield_mid": 13000,
        "yield_half_range": 5000,
        "optimal_ph": 5.5,
        "optimal_temp": 25.0,
        "rainfall_range": (2800, 3600),
        "soil_weights": [0.35, 0.30, 0.15, 0.20],
        "irrigation_weights": [0.10, 0.35, 0.30, 0.25],
        "perennial": False,
    },
}

SOIL_TYPES = ["Laterite", "Red Loam", "Clay Loam", "Sandy Loam"]
IRRIGATION_TYPES = ["Rainfed", "Drip", "Sprinkler", "Canal"]
ZONES = ["Coastal", "Midland", "Malnad"]


def normalize(val, low, high):
    if high == low:
        return 0.5
    return max(0.0, min(1.0, (val - low) / (high - low)))


def ph_score(ph, optimal_ph):
    diff = abs(ph - optimal_ph)
    return max(0.0, 1.0 - diff / 2.0)


def temp_score(temp, optimal_temp):
    diff = abs(temp - optimal_temp)
    return max(0.0, 1.0 - diff / 10.0)


def plant_age_factor(age, max_age=15):
    if age <= 0:
        return 0.0
    if age <= max_age:
        return age / max_age
    plateau = 1.0 - 0.005 * (age - max_age)
    return max(0.7, plateau)


def generate_crop_data(crop_name, config, n_rows):
    data = []
    y_mid = config["yield_mid"]
    y_hr = config["yield_half_range"]
    y_min = config["yield_min"]
    y_max = config["yield_max"]
    opt_ph = config["optimal_ph"]
    opt_temp = config["optimal_temp"]
    rf_low, rf_high = config["rainfall_range"]
    noise_std = y_hr * 0.05

    for _ in range(n_rows):
        soil_type = np.random.choice(SOIL_TYPES, p=config["soil_weights"])
        irrigation_type = np.random.choice(IRRIGATION_TYPES, p=config["irrigation_weights"])

        soil_ph = np.random.uniform(4.5, 6.5)
        soil_nitrogen = np.random.uniform(100, 350)
        soil_phosphorus = np.random.uniform(10, 60)
        soil_potassium = np.random.uniform(80, 300)
        soil_organic_matter = np.random.uniform(0.8, 3.5)

        annual_rainfall = np.random.uniform(rf_low, rf_high)
        temperature_avg = np.random.uniform(22, 32)
        temperature_min = np.random.uniform(18, max(19, temperature_avg - 4))
        temperature_max = np.random.uniform(max(28, temperature_avg + 3), 36)
        humidity = np.random.uniform(65, 92)
        sunshine_hours = np.random.uniform(3.5, 7.5)

        if irrigation_type == "Rainfed":
            irrigation_freq = 0
        elif irrigation_type == "Drip":
            irrigation_freq = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
        elif irrigation_type == "Sprinkler":
            irrigation_freq = np.random.choice([2, 3, 4, 5], p=[0.3, 0.3, 0.25, 0.15])
        else:
            irrigation_freq = np.random.choice([3, 4, 5, 6, 7], p=[0.2, 0.25, 0.25, 0.2, 0.1])

        fertilizer_N = np.random.uniform(60, 200)
        fertilizer_P = np.random.uniform(20, 80)
        fertilizer_K = np.random.uniform(40, 150)
        pesticide = np.random.uniform(0.5, 4.5)
        area = np.random.uniform(0.5, 10.0)
        mechanization = np.random.choice([0, 1], p=[0.4, 0.6])
        crop_variety = np.random.choice([0, 1], p=[0.35, 0.65])

        if config["perennial"]:
            plant_age = np.random.uniform(5, 35)
        else:
            plant_age = 0.0

        elevation = np.random.uniform(0, 950)
        slope = np.random.uniform(0, 35)

        if elevation < 100:
            zone = "Coastal"
        elif elevation < 400:
            zone = "Midland"
        else:
            zone = "Malnad"

        prev_yield = y_mid + np.random.uniform(-y_hr * 0.8, y_hr * 0.8)
        prev_yield = max(y_min, min(y_max, prev_yield))

        rainfall_norm = normalize(annual_rainfall, rf_low, rf_high) - 0.5
        nitrogen_norm = normalize(soil_nitrogen, 100, 350) - 0.5
        phosphorus_norm = normalize(soil_phosphorus, 10, 60) - 0.5
        potassium_norm = normalize(soil_potassium, 80, 300) - 0.5
        organic_norm = normalize(soil_organic_matter, 0.8, 3.5) - 0.5
        fert_n_norm = normalize(fertilizer_N, 60, 200) - 0.5
        fert_p_norm = normalize(fertilizer_P, 20, 80) - 0.5
        fert_k_norm = normalize(fertilizer_K, 40, 150) - 0.5
        ph_s = ph_score(soil_ph, opt_ph) - 0.5
        temp_s = temp_score(temperature_avg, opt_temp) - 0.5
        humidity_norm = normalize(humidity, 65, 92) - 0.5
        sunshine_norm = normalize(sunshine_hours, 3.5, 7.5) - 0.5
        irrig_norm = normalize(irrigation_freq, 0, 7) - 0.5
        area_norm = normalize(area, 0.5, 10.0) - 0.5
        pesticide_norm = normalize(pesticide, 0.5, 4.5) - 0.5
        crop_variety_c = crop_variety - 0.5
        mechanization_c = mechanization - 0.5

        yield_delta = 0.0

        yield_delta += 0.20 * rainfall_norm
        yield_delta += 0.12 * nitrogen_norm
        yield_delta += 0.06 * phosphorus_norm
        yield_delta += 0.08 * potassium_norm
        yield_delta += 0.05 * organic_norm
        yield_delta += 0.12 * fert_n_norm
        yield_delta += 0.05 * fert_p_norm
        yield_delta += 0.06 * fert_k_norm
        yield_delta += 0.08 * ph_s
        yield_delta += 0.06 * temp_s
        yield_delta += 0.04 * humidity_norm
        yield_delta += 0.03 * sunshine_norm
        yield_delta += 0.05 * irrig_norm
        yield_delta += 0.06 * crop_variety_c
        yield_delta += 0.03 * mechanization_c
        yield_delta -= 0.02 * pesticide_norm

        if config["perennial"] and plant_age > 0:
            age_f = plant_age_factor(plant_age) - 0.5
            yield_delta += 0.10 * age_f

        if crop_name == "black pepper":
            elev_norm = normalize(elevation, 100, 950) - 0.5
            slope_norm = normalize(slope, 0, 35) - 0.5
            yield_delta += 0.08 * elev_norm + 0.05 * slope_norm

        if crop_name == "banana":
            yield_delta += 0.08 * irrig_norm + 0.06 * temp_s

        if crop_name == "sweet potato":
            yield_delta += 0.04 * sunshine_norm

        yield_delta = max(-0.8, min(0.8, yield_delta))

        yield_val = y_mid + yield_delta * y_hr * 2.0
        noise = np.random.normal(0, noise_std)
        yield_val += noise
        yield_val = max(y_min, min(y_max, yield_val))

        row = {
            "crop": crop_name,
            "season": config["season"],
            "soil_type": soil_type,
            "soil_pH": round(soil_ph, 2),
            "soil_nitrogen_kg_ha": round(soil_nitrogen, 2),
            "soil_phosphorus_kg_ha": round(soil_phosphorus, 2),
            "soil_potassium_kg_ha": round(soil_potassium, 2),
            "soil_organic_matter_percent": round(soil_organic_matter, 2),
            "annual_rainfall_mm": round(annual_rainfall, 1),
            "temperature_avg_C": round(temperature_avg, 2),
            "temperature_min_C": round(temperature_min, 2),
            "temperature_max_C": round(temperature_max, 2),
            "humidity_percent": round(humidity, 2),
            "sunshine_hours_per_day": round(sunshine_hours, 2),
            "irrigation_type": irrigation_type,
            "irrigation_frequency_days": irrigation_freq,
            "fertilizer_N_applied": round(fertilizer_N, 2),
            "fertilizer_P_applied": round(fertilizer_P, 2),
            "fertilizer_K_applied": round(fertilizer_K, 2),
            "pesticide_usage_kg_ha": round(pesticide, 2),
            "area_under_cultivation_ha": round(area, 2),
            "farm_mechanization": mechanization,
            "crop_variety": crop_variety,
            "plant_age_years": round(plant_age, 1) if config["perennial"] else 0,
            "previous_year_yield": round(prev_yield, 1),
            "elevation_m": round(elevation, 1),
            "slope_percent": round(slope, 2),
            "district_zone": zone,
            "yield_kg_ha": round(yield_val, 2),
        }
        data.append(row)

    return pd.DataFrame(data)


def main():
    os.makedirs("data", exist_ok=True)
    all_dfs = []
    for crop_name, config in CROPS.items():
        print(f"Generating {ROWS_PER_CROP} rows for {crop_name}...")
        df = generate_crop_data(crop_name, config, ROWS_PER_CROP)
        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    output_path = "data/dakshina_kannada_crop_data.csv"
    full_df.to_csv(output_path, index=False)
    print(f"\nDataset saved to {output_path}")
    print(f"Total rows: {len(full_df)}")
    print(f"Columns: {len(full_df.columns)}")
    print(f"\nYield statistics per crop:")
    for crop_name in CROPS:
        subset = full_df[full_df["crop"] == crop_name]
        y = subset["yield_kg_ha"]
        print(f"  {crop_name:15s}: min={y.min():.1f}, max={y.max():.1f}, mean={y.mean():.1f}, std={y.std():.1f}")


if __name__ == "__main__":
    main()

# Crop Yield Prediction System — Dakshina Kannada

An AI-powered deep learning web application for predicting crop yields across eight major crops in the Dakshina Kannada region of Karnataka, India. Built as an MTech-level research-grade project.

## Project Structure

```
crop-yield-prediction/
├── backend/
│   ├── app.py                  # Flask API server (port 5000)
│   ├── dataset_generator.py    # Generates synthetic dataset (16,000+ rows)
│   ├── train_models.py         # Trains 48 DL models (8 crops × 6 models)
│   ├── predict.py              # Inference module with model caching
│   ├── models/                 # Saved .h5 models and .pkl scalers
│   ├── data/
│   │   └── dakshina_kannada_crop_data.csv  # Generated dataset
│   ├── metrics/
│   │   └── model_metrics.json  # Evaluation metrics for all models
│   └── requirements.txt        # Python dependencies
├── frontend/
│   ├── index.html              # Landing / Home page
│   ├── predict.html            # Prediction interface
│   ├── crops.html              # Crop information table & seasonal calendar
│   ├── comparison.html         # Model comparison with 5 Chart.js charts
│   ├── css/
│   │   └── style.css           # Custom academic dashboard styling
│   ├── js/
│   │   ├── predict.js          # Dynamic form rendering & prediction
│   │   ├── crops.js            # Crop table & calendar rendering
│   │   └── comparison.js       # Charts, sorting, recommendations
│   └── assets/
└── README.md
```

## Setup Instructions

### Prerequisites
- Python 3.9+ with pip
- Modern web browser (Chrome, Firefox, Edge)

### Step 1: Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Step 2: Generate the Dataset
```bash
python dataset_generator.py
```
This generates `data/dakshina_kannada_crop_data.csv` with 17,600 rows (2,200 per crop × 8 crops) of statistically realistic synthetic data reflecting the agronomic and climatic conditions of Dakshina Kannada.

### Step 3: Train All Models
```bash
python train_models.py
```
This trains 48 deep learning models (8 crops × 6 architectures) and saves them to `models/`. Training takes approximately 30–60 minutes depending on hardware (GPU recommended). Use `--retrain` flag to force retraining of existing models.

### Step 4: Start the Flask API Server
```bash
python app.py
```
The API will start on `http://localhost:5000`.

### Step 5: Open the Frontend
Open `frontend/index.html` in your browser, or use a local server:
```bash
cd frontend
python -m http.server 8080
```
Then navigate to `http://localhost:8080/index.html`.

## Deep Learning Models

### 1. LSTM (Long Short-Term Memory)
A recurrent neural network architecture with memory cells that capture temporal dependencies in feature patterns. The model uses two stacked LSTM layers (128 and 64 units) with dropout regularization to prevent overfitting. Well-suited for tabular data where feature interactions have sequential characteristics.

### 2. BiLSTM (Bidirectional LSTM)
Extends LSTM by processing input sequences in both forward and backward directions, capturing context from the entire feature space simultaneously. Includes batch normalization for training stability. Often achieves higher accuracy than unidirectional LSTM due to bidirectional feature learning.

### 3. GRU (Gated Recurrent Unit)
A simplified recurrent architecture with fewer gates than LSTM, offering faster training while maintaining comparable performance. Uses the same two-layer stacked design (128 and 64 units) with dropout. Particularly effective when the dataset size is moderate.

### 4. CNN-LSTM Hybrid
Combines 1D convolutional layers for local feature extraction with LSTM layers for pattern sequencing. The Conv1D layers (64 and 32 filters) identify spatial correlations among input features, which are then fed to LSTM for temporal pattern learning. Effective for capturing both local and global feature relationships.

### 5. Transformer (Attention-based)
Adapts the TabTransformer architecture for regression tasks. Uses multi-head self-attention (4 heads, key dimension 16) to learn feature interactions without sequential assumptions. Includes feed-forward networks and layer normalization. Best at capturing complex non-linear relationships across all features simultaneously.

### 6. Stacked Autoencoder + Regression
A two-stage approach: first, a deep autoencoder (128→64→32→64→128) is pretrained for 50 epochs to learn compressed feature representations through reconstruction. Then the encoder is retained and a regression head (32→16→1) is added and fine-tuned for 100 epochs. The autoencoder bottleneck acts as a learned feature selector.

## Feature Descriptions

| Feature | Type | Range | Description |
|---|---|---|---|
| soil_type | Categorical | Laterite, Red Loam, Clay Loam, Sandy Loam | Dominant soil classification |
| soil_pH | Numeric | 4.5 – 6.5 | Soil acidity level (DK soils are acidic) |
| soil_nitrogen_kg_ha | Numeric | 100 – 350 | Available nitrogen in soil |
| soil_phosphorus_kg_ha | Numeric | 10 – 60 | Available phosphorus in soil |
| soil_potassium_kg_ha | Numeric | 80 – 300 | Available potassium in soil |
| soil_organic_matter_percent | Numeric | 0.8 – 3.5 | Organic matter content |
| annual_rainfall_mm | Numeric | 2,800 – 4,500 | Total annual rainfall |
| temperature_avg_C | Numeric | 22 – 32 | Average annual temperature |
| temperature_min_C | Numeric | 18 – 24 | Minimum growing season temperature |
| temperature_max_C | Numeric | 28 – 36 | Maximum growing season temperature |
| humidity_percent | Numeric | 65 – 92 | Average relative humidity |
| sunshine_hours_per_day | Numeric | 3.5 – 7.5 | Daily sunshine hours |
| irrigation_type | Categorical | Rainfed, Drip, Sprinkler, Canal | Primary irrigation method |
| irrigation_frequency_days | Numeric | 0 – 7 | Days between irrigation cycles |
| fertilizer_N_applied | Numeric | 60 – 200 | Nitrogen fertilizer (kg/ha) |
| fertilizer_P_applied | Numeric | 20 – 80 | Phosphorus fertilizer (kg/ha) |
| fertilizer_K_applied | Numeric | 40 – 150 | Potassium fertilizer (kg/ha) |
| pesticide_usage_kg_ha | Numeric | 0.5 – 4.5 | Pesticides applied per hectare |
| area_under_cultivation_ha | Numeric | 0.5 – 10 | Total cultivation area |
| farm_mechanization | Categorical | Manual (0), Mechanized (1) | Farm mechanization level |
| crop_variety | Categorical | Traditional (0), HYV (1) | High yielding vs traditional variety |
| plant_age_years | Numeric | 0 – 35 | Age of perennial plants (0 for seasonal) |
| previous_year_yield | Numeric | Varies by crop | Lagged yield from prior year |
| elevation_m | Numeric | 0 – 950 | Elevation above sea level |
| slope_percent | Numeric | 0 – 35 | Land slope percentage |
| district_zone | Categorical | Coastal, Midland, Malnad | Agro-climatic zone of DK |

## Crops Covered

| Crop | Yield Range (kg/ha) | Season | Primary Zones |
|---|---|---|---|
| Rice | 2,800 – 4,500 | Kharif (Jun–Nov) | Coastal, Midland |
| Coconut | 9,000 – 14,000 (nuts/ha) | Perennial | Coastal, Midland |
| Arecanut | 1,800 – 3,200 | Perennial | Midland, Malnad |
| Banana | 25,000 – 40,000 | Annual | All zones |
| Black Pepper | 800 – 2,200 | Perennial | Malnad, Midland |
| Cashew | 600 – 1,800 | Perennial | Coastal |
| Cocoa | 700 – 1,500 | Perennial | Midland, Malnad |
| Sweet Potato | 8,000 – 18,000 | Rabi (Oct–Feb) | Coastal, Midland |

## Evaluation Metrics

All models are evaluated using five standard regression metrics:

- **R² Score (Coefficient of Determination)**: Proportion of variance in yield explained by the model. Target: ≥ 0.85
- **RMSE (Root Mean Square Error)**: Standard deviation of prediction errors, in original units (kg/ha)
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual yield
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error; used to compute Accuracy %
- **NSE (Nash-Sutcliffe Efficiency)**: Measures how well predictions match observations relative to the mean; ranges from -∞ to 1 (perfect)
- **Accuracy %**: Computed as (1 - MAPE/100) × 100

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/predict` | POST | Submit features for yield prediction |
| `/api/metrics` | GET | Retrieve all model evaluation metrics |
| `/api/crops` | GET | Get crop metadata and descriptions |
| `/api/feature-config/<crop>` | GET | Get feature ranges and defaults for a crop |
| `/api/comparison-data` | GET | Get aggregated data for comparison charts |

## License

This project is developed for academic research purposes as part of an MTech program.

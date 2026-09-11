# FloodLens Data

This directory holds the datasets used to train and evaluate FloodLens.

## Dataset strategy

FloodLens prefers the **real, openly licensed** Kaggle dataset
*"Flood Risk in India"* (CC0 / Public Domain), by WARNER / s3programmer.

- Source: https://www.kaggle.com/datasets/s3programmer/flood-risk-in-india
- License: CC0 Public Domain (no copyright restrictions)
- Download size: ~850 KB (zipped CSV)

The dataset contains ~50,000 rows of Indian locations with meteorological,
hydrological, geographical and socio-economic features, plus a binary
`Flood_Occurred` label. The training pipeline derives a 4-class
**flood risk level** (Low / Moderate / High / Critical) from a risk score that
combines the environmental drivers and flood occurrence; the method is
documented in `docs/PROJECT_REPORT.md`.

If the real dataset cannot be downloaded (e.g. no Kaggle credentials,
offline environment, classroom demo), the training pipeline automatically
falls back to a **synthetic dataset** of the same schema, generated with
physically plausible correlations and clearly labelled as synthetic in
model metadata and the UI. The synthetic data is only for demonstrating
the pipeline and UI — it MUST NOT be presented as real-world data.

## Directory layout

```
data/
├── raw/              # Original CSV(s), untouched
├── processed/        # Cleaned, feature-engineered training data
└── synthetic/        # Synthetic fallback data (generated)
```

## How to obtain the real dataset

1. Sign in to Kaggle (free account).
2. Visit https://www.kaggle.com/datasets/s3programmer/flood-risk-in-india
3. Download the ZIP and extract `flood_risk_dataset_india.csv` (or similar CSV)
   into `data/raw/`, renaming it to `flood_risk_india.csv`.
4. Run `python scripts/train_pipeline.py` — it will detect the file and
   train on real data.

If the file is not present, the pipeline will generate the synthetic
dataset automatically.

## Data dictionary (real dataset)

| Column                   | Type   | Description |
|--------------------------|--------|-------------|
| Latitude                 | float  | Latitude of observation point |
| Longitude                | float  | Longitude of observation point |
| Rainfall_mm              | float  | Rainfall recorded (mm) |
| Temperature_C            | float  | Temperature (°C) |
| Humidity_pct             | float  | Relative humidity (%) |
| River_Discharge_m3_s     | float  | River discharge (m³/s) |
| Water_Level_m            | float  | River water level (m) |
| Elevation_m              | float  | Elevation above sea level (m) |
| Land_Cover               | cat    | Urban / Forest / Agricultural / Water Body / Desert |
| Soil_Type                | cat    | Sandy / Clay / Loam / Silt / Peat |
| Population_Density       | float  | Persons per km² |
| Infrastructure           | 0/1    | Presence of flood-control infrastructure |
| Historical_Floods        | 0/1    | Historical flood occurrence for that location |
| Flood_Occurred           | 0/1    | Binary target (1 = flood event) |

## Engineered risk target (4 classes)

We compute a continuous **Risk Score** from normalized Rainfall, River
Discharge, Water Level, Humidity and the Historical_Floods indicator, then
bin it into four ordinal classes, further informed by `Flood_Occurred`.
See `src/feature_engineering.py` for the exact formula and rationale.

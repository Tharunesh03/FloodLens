# FloodLens — AI-Based Flood Risk Prediction and Early Warning System

## A College Data-Science Project Report

---

## 1. Title

**FloodLens: AI-Based Flood Risk Prediction and Early Warning System**

## 2. Abstract

Floods are among the most frequent and destructive natural hazards,
affecting millions of people each year. Early, data-driven assessment of
flood risk can improve community preparedness and response planning. This
project presents **FloodLens**, a machine-learning-based flood risk
estimation system that classifies a location's flood risk into four levels
— Low, Moderate, High and Critical — using environmental, hydrological,
geographical and historical features. FloodLens is implemented as a Flask
web application with a JSON API, an interactive dashboard, a prediction
history store (SQLite) and a reproducible training pipeline. The pipeline
covers data cleaning, exploratory data analysis, feature engineering,
multi-model comparison, hyperparameter tuning, explainability and
deployment. A public-domain dataset on flood risk in India is used by
default; if the real dataset is unavailable, a clearly-labelled synthetic
fallback dataset is generated so the full stack remains demonstrable. The
best-performing model in our experiments (Logistic Regression with
balanced class weights) achieves a macro-F1 score of approximately **0.78**
on the held-out synthetic test set, with macro-averaged recall of
**0.81**, showing that simple, transparent models can be competitive for
tabular risk classification when features are engineered appropriately.
The system is designed as an educational decision-support aid; it is not
an official emergency warning.

## 3. Introduction

Flooding causes loss of life, damage to infrastructure, disruption of
supply chains, and long-term economic harm, particularly in densely
populated, low-lying regions. While government meteorological and
hydrological agencies issue authoritative warnings, there is educational
and practical value in transparent, multi-factor ML models that can
explain *why* a risk estimate is elevated. FloodLens is built as a
complete data-science project that showcases every stage of a modern ML
workflow and packages the result as a runnable, deployable web
application.

## 4. Background

Flood risk arises from a combination of meteorological drivers (rainfall
intensity and duration), hydrological conditions (river discharge, water
level, antecedent soil moisture), geography (elevation, slope, proximity
to water bodies), surface characteristics (land cover, soil drainage),
built infrastructure (levees, drainage networks) and historical
exposure. Single-variable thresholds (e.g., "more than X mm of rain
implies risk") fail to capture these interactions and produce many false
alarms or missed events. Multi-factor ML is therefore a natural approach
to risk stratification.

## 5. Problem Statement

Design and implement a complete, deployable ML system that:

- Accepts environmental/hydrological inputs for a location.
- Outputs a classified flood risk level (Low / Moderate / High / Critical).
- Quantifies uncertainty via class probabilities.
- Explains which factors most influenced the prediction.
- Exposes both a web UI and a JSON API.
- Is accompanied by EDA, model comparison, tuning, tests and documentation.
- Clearly disclaims that it is not an official emergency warning.

## 6. Existing System

Many existing flood-warning systems are operated by national weather and
water agencies using physically-based hydrological models and large
sensor/forecast networks. These are authoritative but:

- Often lack open APIs suitable for educational projects.
- Require large computational and data resources.
- Offer limited transparency into how individual inputs combine into a
  risk score.
- Are not easily portable or reproducible for a college portfolio.

Many Kaggle/ML flood projects are notebook-only, without a deployed app,
tests, or a clean codebase. FloodLens aims to fill that gap.

## 7. Limitations of Existing Systems

- **Threshold-only alerts** produce too many false alarms.
- **Black-box models** (e.g., deep learning on remote sensing) can be
  hard to explain to end users.
- **Notebook-only projects** are not deployable or testable.
- **Synthetic-data claims** in some educational demos are not always
  disclosed, which is scientifically dishonest.

## 8. Proposed System

FloodLens is a Flask-based Python web application with:

1. A modular training pipeline (`src/`) that loads, cleans, engineers
   features for, and models flood risk.
2. A prediction service that wraps the trained model and provides
   lightweight explanations.
3. A responsive UI with prediction form, results, dashboard, history,
   methodology and API pages.
4. A JSON API (`/api/predict`, `/api/health`, `/api/metadata`).
5. SQLite storage for prediction history.
6. Automated tests with pytest.
7. Deployment files for free hosting (Render).

## 9. Objectives

- Build a 4-class flood risk classifier using multiple meaningful features.
- Compare multiple ML algorithms and tune the best.
- Provide per-prediction explanations.
- Build a production-shaped Flask app with API and dashboard.
- Persist prediction history in SQLite.
- Document dataset, methodology, limitations and ethics transparently.
- Provide a complete test suite and deployment path.

## 10. Scope

- **In scope:** tabular classification, web UI, API, explainability,
  dashboard, history, tests, deployment on free hosting.
- **Out of scope** (future work): real-time weather API ingestion,
  satellite or GIS integration, deep-learning time-series models, mobile
  apps, operational emergency integration.

## 11. Literature Review

- Classical statistical flood-frequency analysis uses Gumbel/Log-Pearson
  distributions on historical discharge data.
- Machine-learning approaches (Random Forest, Gradient Boosting, SVM)
  have been widely applied to flood susceptibility mapping (e.g., Tehrany
  et al., 2014; Choubin et al., 2019).
- Interpretable models and SHAP/LIME explanations are increasingly
  recommended for hazard modelling to maintain trust.
- Scikit-learn pipelines are recognized best practice to prevent data
  leakage during training and inference.

## 12. Dataset

The default dataset is the **CC0-licensed "Flood Risk in India"** on
Kaggle, containing ~50,000 rows with meteorological, hydrological,
geographical, land-cover, soil and historical-flood fields across India.
When that dataset is not present, the pipeline generates a **synthetic
fallback** with matching schema and physically plausible correlations.
All synthetic outputs are clearly labelled.

## 13. Data Collection

- Real data: download from Kaggle and place in `data/raw/flood_risk_india.csv`.
- Synthetic data: generated programmatically in `src/data_processing.py`
  using numpy with correlations calibrated so models can learn meaningful
  patterns without fabrication claims.

## 14. Data Preprocessing

- Column-name normalization, type coercion.
- Duplicate row removal.
- Physical clipping (e.g., rainfall ≤ 2000 mm).
- Median/mode imputation.
- Categorical canonicalization (case normalization).

## 15. Exploratory Data Analysis

EDA includes:

- Target / risk distribution.
- Rainfall vs risk (box plots).
- Feature correlation heatmap.
- Scatter of river discharge vs water level.
- Class balance statistics.

Charts are rendered with Plotly on the dashboard and saved as PNGs in
`app/static/exports/`.

## 16. Feature Engineering

Engineered features:

- `Rainfall_72h_proxy` (√rainfall × 4.5) as a proxy for antecedent
  wetness.
- `Rainfall_x_Level` interaction between rainfall and water level.
- `Discharge_x_Humidity` interaction.
- `Elevation_inv` = 1/(elevation + 10), representing low-lying risk.

Target construction: a continuous Risk Score (0–1) is computed from
percentiled drivers plus penalties for soil/land cover and mitigation
for infrastructure, then binned into four classes. All observed flood
rows are promoted to at least High.

## 17. System Architecture

```
Browser / API client
        ↓
     Flask app  ←→ SQLite (prediction history)
        ↓
   Prediction service (validation → features → model → explanation)
        ↓
  Joblib-saved Pipeline (preprocessor + classifier)
        ↑
   Training pipeline (data → clean → features → tune → save)
```

## 18. Methodology

1. Data loading and cleaning.
2. Feature engineering and target construction.
3. Stratified train/val/test split (70/15/15).
4. Preprocessing inside a scikit-learn `Pipeline` (median imputation +
   scaling for numerics, one-hot for categoricals).
5. Train Logistic Regression, Decision Tree, Random Forest, Gradient
   Boosting; optionally XGBoost.
6. Select best model by macro-F1 with recall tie-break.
7. Lightweight hyperparameter tuning with `RandomizedSearchCV` (3-fold
   stratified CV).
8. Evaluate on held-out test set using accuracy, macro precision/recall/F1,
   weighted F1 and confusion matrix.
9. Serialize the pipeline with joblib; save metadata JSON.
10. At serving time, load the model once and generate predictions with
    explanations.

### Why stratified splitting?

The dataset does not contain a reliable timestamp column for chronological
splitting; we therefore use stratified random splits to preserve class
balance across folds.

## 19. Machine Learning Algorithms

- **Logistic Regression:** linear baseline with balanced class weights;
  fast, interpretable.
- **Decision Tree:** non-linear baseline; prone to overfitting.
- **Random Forest:** ensemble of trees, robust but heavier.
- **Gradient Boosting:** sequential ensemble, strong on tabular data.
- **XGBoost:** optional optimized gradient boosting.

## 20. Model Training

The pipeline uses `class_weight='balanced'` for applicable models to
counter class imbalance. Preprocessing is wrapped in a Pipeline to
guarantee that scaling/encoding is fit only on training data.

## 21. Hyperparameter Tuning

The best model (Logistic Regression) is tuned via 3-fold stratified
`RandomizedSearchCV` over regularization strength `C` using macro-F1.
Best parameters found on the synthetic data: `C=1.0`, best CV macro-F1
≈ 0.77.

## 22. Model Evaluation

Held-out test-set metrics (synthetic data):

| Metric            | Value  |
|-------------------|--------|
| Accuracy          | 0.757  |
| Precision (macro) | 0.759  |
| Recall (macro)    | **0.811** |
| F1 (macro)        | **0.779** |
| F1 (weighted)     | 0.752  |

Confusion matrix and per-class precision/recall/F1 are viewable on the
dashboard.

## 23. Results

- A transparent Logistic Regression baseline with engineered features
  matches or outperforms heavier tree ensembles on this dataset.
- Recall for high-risk classes is prioritized; the system leans toward
  flagging risk rather than missing it.
- The interactive dashboard makes model behaviour accessible to
  non-technical viewers.
- The JSON API supports integration with other tools.

## 24. Explainability

For every prediction the UI reports:

- Predicted risk level and confidence.
- Per-class probability bars.
- Top 4 contributing features (e.g., "Major factor elevates risk from
  water level", "Significant factor reduces risk via higher elevation"),
  each labelled by magnitude (Minor / Notable / Significant / Major).

Contributions are derived from z-scores relative to training data for
numerics, and category-vs-global risk means for categorics. They are
explicitly described as influences on the model, not proven causes.

## 25. Web Application

Pages: Home, Predict, Dashboard, History, Methodology, API docs, About.
A responsive CSS theme (light/dark aware) and vanilla JS deliver a clean,
dashboard-like experience. Forms use Flask-WTF with CSRF protection.
Prediction history is stored in SQLite and can be cleared.

## 26. API

- `GET /api/health` — liveness.
- `POST /api/predict` — prediction with JSON input/output.
- `GET /api/metadata` — model metadata.
- All endpoints return proper HTTP codes (200/400/503).

## 27. Testing

Automated tests (`pytest`) cover:

- Synthetic data generation.
- Cleaning (outliers, missing values).
- Feature engineering (target construction, class constraints).
- Input validation.
- End-to-end prediction.
- Invalid categorical values.
- Health, predict (valid + invalid), metadata API endpoints.
- Page loads for home, predict, dashboard, about.

All 16 tests pass.

## 28. Deployment

Render is recommended as a free host. See `README.md` for step-by-step
instructions. `Procfile`, `render.yaml` and `runtime.txt` are included.
Gunicorn serves the Flask WSGI app.

## 29. Limitations

- Geographic coverage is India; model may not generalize globally.
- Without timestamps, no proper time-series validation is done.
- The 4-class target is derived, not observed severity.
- Synthetic fallback data is for demonstration only.
- No real-time sensor/weather integration.
- Class imbalance and distribution shift can degrade performance.
- Free-host SQLite storage is ephemeral.

## 30. Ethical Considerations

- Not an official warning; never override government guidance.
- False negatives can be dangerous — recall is prioritized but imperfect.
- No personal data is collected.
- Dataset source and license are clearly documented.
- Synthetic data is always labelled.

## 31. Future Enhancement

- Real-time weather/API ingestion.
- Satellite / GIS integration with Leaflet + OpenStreetMap.
- LSTM/Temporal Fusion Transformer time-series models.
- IoT sensor integration.
- Continuous model monitoring and retraining.
- Multi-region training.
- Docker/Kubernetes deployment, mobile apps, multi-channel alerts.

## 32. Conclusion

FloodLens demonstrates a complete, reproducible and deployable
data-science solution for flood-risk estimation. It balances performance
with interpretability, documents its data and limitations honestly, and
ships as a production-shaped Flask application suitable for a college
capstone, a GitHub portfolio, or a starting point for further research.

## 33. References

1. Flood Risk in India dataset (CC0). Kaggle. https://www.kaggle.com/datasets/s3programmer/flood-risk-in-india
2. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR.
3. Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
4. Flask documentation. https://flask.palletsprojects.com/
5. Global Flood Awareness System (GloFAS). https://open-meteo.com/en/docs/flood-api
6. Tehrany et al. (2014). Flood susceptibility mapping using a novel ensemble.
7. Choubin et al. (2019). An ensemble approach to flood susceptibility modelling.

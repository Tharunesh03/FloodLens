# FloodLens — Presentation Deck (15 slides)

Use this as speaker notes for a 12–15 minute presentation. Each slide is
self-contained; use the dashboard screenshots in `app/static/exports/` as
visuals.

---

## Slide 1 — Title

**FloodLens: AI-Based Flood Risk Prediction & Early Warning**

- Subtitle: An end-to-end machine-learning decision-support system
- Your name, course/college, date
- One-line visual: flood/water-themed logo from
  `app/static/images/favicon.svg`

---

## Slide 2 — Problem

- Floods are among the most destructive natural hazards globally.
- Single-variable thresholds (rainfall > X mm) produce many false
  alarms / misses.
- Stakeholders need transparent, multi-factor risk estimates with
  explanations.
- Most educational ML projects stop at notebooks; we need a deployable
  system.

---

## Slide 3 — Motivation

- Showcase the full data-science pipeline end to end.
- Use real, openly licensed data; provide a synthetic fallback when
  offline.
- Prioritize recall for high-risk events.
- Deliver a production-shaped Flask app with API and dashboard.
- Be honest about limitations and ethics.

---

## Slide 4 — Existing Systems & Gaps

- Government systems: authoritative, but not portable/reproducible for a
  student portfolio.
- Notebooks on Kaggle: rarely deployed, lack tests, often hide data
  limitations.
- Simple rules: ignore interactions between drivers.
- Gap: a complete, tested, explainable, deployable educational project.

---

## Slide 5 — Proposed System: FloodLens

- Classifies flood risk into **Low / Moderate / High / Critical**.
- Inputs: rainfall, temperature, humidity, river discharge, water level,
  elevation, soil type, land cover, infrastructure, historical floods.
- Outputs: predicted class, per-class probabilities, confidence,
  contributing factors, precautions.
- Ships as a responsive Flask web app + JSON API + interactive dashboard
  + SQLite history.

---

## Slide 6 — Objectives

1. Build a multi-factor flood risk classifier.
2. Compare multiple ML algorithms and tune the best.
3. Provide per-prediction explanations.
4. Build a usable web UI and JSON API.
5. Persist prediction history in SQLite.
6. Cover code with automated tests.
7. Deploy for free on a public host.

---

## Slide 7 — Dataset

- **Preferred:** *Flood Risk in India* (Kaggle), CC0 Public Domain, ~50k rows.
- Features: meteorological, hydrological, geographical, soil, land cover,
  infrastructure, historical floods, binary flood flag.
- **Fallback:** a clearly-labelled synthetic dataset generated with
  physically plausible correlations when the real CSV is absent.
- Coverage: India. Documented in `data/README.md`.

---

## Slide 8 — Methodology

Pipeline visual:

```
Data → Cleaning → Feature Engineering → Model Training & Tuning
       → Risk Prediction → Explainability → UI + API
```

- Cleaning: de-duplication, clipping, imputation, type coercion.
- Features: rainfall/water-level/discharge interactions, elevation
  proxy, risk score → 4 ordinal classes.
- Preprocessing inside a scikit-learn Pipeline (median imputation +
  scaling for numerics, one-hot for categorics).
- Stratified 70/15/15 split; RandomizedSearchCV (3-fold CV) for tuning.

---

## Slide 9 — ML Models Compared

1. Logistic Regression (balanced class weights)
2. Decision Tree
3. Random Forest (200 trees)
4. Gradient Boosting (150 estimators)
5. XGBoost (optional)

Selection criterion: **macro-F1** with recall tie-break — because missing
high-risk events costs more than false alarms.

---

## Slide 10 — Results

(These numbers reflect the synthetic fallback demo; rerun on the real
dataset for real-data metrics.)

- Selected model: **Logistic Regression** (best macro-F1 & recall on
  validation, fastest inference, interpretable).
- Test-set metrics:
  - Accuracy ≈ 0.76
  - Macro Precision ≈ 0.76
  - **Macro Recall ≈ 0.81**
  - **Macro F1 ≈ 0.78**
- Show confusion matrix, per-class precision/recall/F1 from the
  dashboard.

---

## Slide 11 — Application Demo

Walk through the live app (screenshots if live demo is unavailable):

- **/predict** — enter environmental inputs, click "Estimate Risk".
- Result: risk badge, probability bars, contributing factors, precautions.
- **/dashboard** — risk distribution, correlation heatmap, rainfall box
  plots, scatter, feature importance, confusion matrix, model comparison.
- **/history** — recent predictions stored in SQLite.
- **/api** — try `/api/health` and `/api/predict` directly.

---

## Slide 12 — Architecture

- Frontend: vanilla HTML/CSS/JS + Jinja templates (Plotly for charts).
- Backend: Python 3.11 + Flask, CSRF protection via Flask-WTF.
- ML: scikit-learn Pipeline saved with joblib; model loaded once.
- Storage: SQLite (prediction history only).
- Deployment: gunicorn behind Render free tier (Procfile + render.yaml).
- Tests: pytest (16 tests covering data, prediction, API, pages).

---

## Slide 13 — Limitations & Ethics

- **Not an official warning** — always follow government guidance.
- Geographic coverage: India (real data) / synthetic for demo.
- No time-series / sensor integration yet.
- 4-class target is derived from a risk score, not observed severity.
- Synthetic data must not be presented as real-world evidence.
- Recall is prioritized but not perfect — false negatives are possible.
- Free-tier SQLite storage is ephemeral.

---

## Slide 14 — Future Scope

- Real-time weather API ingestion (Open-Meteo, IMD, NOAA).
- Spatiotemporal models (LSTM/Temporal Fusion Transformers).
- GIS integration (Leaflet + OpenStreetMap).
- IoT water-level sensors.
- Continuous monitoring + retraining.
- Multi-region datasets; mobile app; alerting (email/SMS).

---

## Slide 15 — Conclusion

- FloodLens is a complete, reproducible and deployable data-science
  project.
- It demonstrates the full ML lifecycle with honest documentation and a
  usable UI/API.
- It is suitable as a college capstone, GitHub portfolio piece, or
  starting point for further flood-risk research.
- Thank you — questions?

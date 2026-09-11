"""
Generate Plotly JSON charts used by the FloodLens dashboard, plus a small
set of static PNG EDA outputs saved to app/static/exports.

All charts are generated lazily (once) and cached in memory when the Flask
app starts.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config, RISK_LEVELS, RISK_COLORS
from src.feature_engineering import RISK_LABELS

logger = logging.getLogger("floodlens.viz")


def _load_engineered() -> Optional[pd.DataFrame]:
    p = Config.PROCESSED_DATA_DIR / "engineered.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def _load_metadata() -> Optional[Dict[str, Any]]:
    mp = Config.MODELS_DIR / Config.METADATA_FILE
    if not mp.exists():
        return None
    with open(mp) as f:
        return json.load(f)


def _fig_to_json(fig) -> str:
    return pio.to_json(fig)


def generate_dashboard_charts() -> Dict[str, Any]:
    """Return a dict of chart JSON strings + summary stats for the dashboard."""
    Config.ensure_dirs()
    df = _load_engineered()
    meta = _load_metadata()
    charts: Dict[str, Any] = {"available": df is not None and meta is not None}

    if df is None or meta is None:
        return charts

    color_seq = [RISK_COLORS[lbl] for lbl in RISK_LABELS]

    # 1. Risk distribution
    counts = df["Risk_Level"].value_counts().reindex(RISK_LABELS).fillna(0)
    fig = go.Figure(data=[go.Bar(
        x=counts.index, y=counts.values,
        marker_color=[RISK_COLORS[l] for l in counts.index],
        text=counts.values.astype(int), textposition="outside",
    )])
    fig.update_layout(
        title="Risk Level Distribution",
        xaxis_title="Risk Level", yaxis_title="Number of samples",
        template="plotly_white", height=360, margin=dict(l=40, r=20, t=50, b=40),
    )
    charts["risk_dist"] = _fig_to_json(fig)

    # 2. Rainfall vs Risk
    fig = px.box(
        df, x="Risk_Level", y="Rainfall_mm",
        category_orders={"Risk_Level": RISK_LABELS},
        color="Risk_Level", color_discrete_map=RISK_COLORS,
        title="Rainfall Distribution by Risk Level",
    )
    fig.update_layout(template="plotly_white", height=360, showlegend=False,
                      margin=dict(l=40, r=20, t=50, b=40))
    charts["rainfall_box"] = _fig_to_json(fig)

    # 3. Correlation heatmap (numeric features)
    num_cols = [c for c in [
        "Rainfall_mm", "Temperature_C", "Humidity_pct",
        "River_Discharge_m3_s", "Water_Level_m", "Elevation_m",
        "Population_Density", "Risk_Score",
    ] if c in df.columns]
    corr = df[num_cols].corr().round(2)
    fig = px.imshow(
        corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
        title="Feature Correlation (Pearson)", range_color=[-1, 1],
    )
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=40, r=20, t=50, b=80))
    charts["corr_heatmap"] = _fig_to_json(fig)

    # 4. Water level vs river discharge scatter colored by risk
    sample = df.sample(min(4000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="River_Discharge_m3_s", y="Water_Level_m",
        color="Risk_Level", color_discrete_map=RISK_COLORS,
        category_orders={"Risk_Level": RISK_LABELS},
        title="River Discharge vs Water Level (sampled)",
        opacity=0.6,
    )
    fig.update_layout(template="plotly_white", height=400, margin=dict(l=40, r=20, t=50, b=40))
    charts["scatter"] = _fig_to_json(fig)

    # 5. Feature importance (bar)
    importances = meta.get("feature_importances")
    feat_names = meta.get("feature_names_transformed") or []
    if importances and feat_names and len(importances) == len(feat_names):
        pairs = sorted(zip(feat_names, importances), key=lambda kv: kv[1], reverse=True)[:15]
        labels = [p[0] for p in pairs][::-1]
        vals = [p[1] for p in pairs][::-1]
        fig = go.Figure(data=[go.Bar(x=vals, y=labels, orientation="h")])
        fig.update_layout(
            title=f"Top 15 Feature Importances ({meta['model_selection']['selected_model']})",
            xaxis_title="Importance", template="plotly_white", height=420,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        charts["feature_importance"] = _fig_to_json(fig)
    else:
        charts["feature_importance"] = None

    # 6. Confusion matrix
    cm = np.array(meta["confusion_matrix"])
    fig = px.imshow(
        cm, x=RISK_LABELS, y=RISK_LABELS, text_auto=True,
        color_continuous_scale="Blues", title="Confusion Matrix (Test Set)",
        labels=dict(x="Predicted", y="Actual"),
    )
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=40, r=20, t=50, b=40))
    charts["confusion_matrix"] = _fig_to_json(fig)

    # 7. Model comparison (val f1_macro)
    candidates = meta["model_selection"]["candidates"]
    names = list(candidates.keys())
    f1_vals = [candidates[n]["f1_macro"] for n in names]
    recall_vals = [candidates[n]["recall_macro"] for n in names]
    acc_vals = [candidates[n]["accuracy"] for n in names]
    fig = go.Figure(data=[
        go.Bar(name="F1 (macro)", x=names, y=f1_vals, marker_color="#0ea5e9"),
        go.Bar(name="Recall (macro)", x=names, y=recall_vals, marker_color="#f97316"),
        go.Bar(name="Accuracy", x=names, y=acc_vals, marker_color="#22c55e"),
    ])
    fig.update_layout(barmode="group", title="Model Comparison (Validation Set)",
                      yaxis_title="Score", template="plotly_white", height=400,
                      margin=dict(l=40, r=20, t=50, b=40))
    charts["model_compare"] = _fig_to_json(fig)

    # Summary statistics
    charts["summary"] = {
        "rows": int(len(df)),
        "features": int(len(meta["dataset"]["feature_names"])),
        "model_name": meta["model_selection"]["selected_model"],
        "test_metrics": meta["test_metrics"],
        "class_distribution": meta["dataset"]["class_distribution"],
        "is_synthetic": meta["is_synthetic_data"],
        "trained_at": meta["trained_at"],
        "model_version": meta["model_version"],
    }

    # Save a few static PNGs for docs
    _save_static_pngs(df, meta)

    return charts


def _save_static_pngs(df: pd.DataFrame, meta: Dict[str, Any]):
    out = Config.EXPORTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Risk distribution PNG
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df["Risk_Level"].value_counts().reindex(RISK_LABELS).fillna(0)
    ax.bar(counts.index, counts.values, color=[RISK_COLORS[l] for l in counts.index])
    ax.set_title("Risk Level Distribution")
    ax.set_ylabel("Number of samples")
    fig.tight_layout()
    fig.savefig(out / "risk_distribution.png", dpi=130)
    plt.close(fig)

    # Correlation heatmap PNG
    num_cols = [c for c in [
        "Rainfall_mm", "Temperature_C", "Humidity_pct",
        "River_Discharge_m3_s", "Water_Level_m", "Elevation_m",
        "Population_Density", "Risk_Score",
    ] if c in df.columns]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="RdBu_r", center=0, ax=ax, fmt=".2f")
    ax.set_title("Feature Correlation")
    fig.tight_layout()
    fig.savefig(out / "correlation_heatmap.png", dpi=130)
    plt.close(fig)

    # Confusion matrix PNG
    cm = np.array(meta["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=RISK_LABELS, yticklabels=RISK_LABELS, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix ({meta['model_selection']['selected_model']})")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    # Feature importance PNG
    importances = meta.get("feature_importances")
    feat_names = meta.get("feature_names_transformed") or []
    if importances and feat_names and len(importances) == len(feat_names):
        pairs = sorted(zip(feat_names, importances), key=lambda kv: kv[1], reverse=True)[:15]
        labels = [p[0] for p in pairs][::-1]
        vals = [p[1] for p in pairs][::-1]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(labels, vals, color="#0ea5e9")
        ax.set_title(f"Top Feature Importances ({meta['model_selection']['selected_model']})")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(out / "feature_importance.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    charts = generate_dashboard_charts()
    print("Dashboard charts generated:", list(charts.keys()))

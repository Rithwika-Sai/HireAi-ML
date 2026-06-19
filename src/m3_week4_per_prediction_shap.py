"""
M3 Week 4 — Per-prediction SHAP for the /explain API, in GenAI G3's agreed schema.

For sample (candidate, job) pairs, loads the saved fit classifier and produces:
  predicted_label, probabilities, base_value,
  top_positive_drivers / top_negative_drivers (top 3 each) — with signed
  shap_value AND raw feature_value (per G3's sign-off).

Writes reports/shap/per_prediction_examples.json.
Run:  python m3_week4_per_prediction_shap.py
"""

import os
import json

import numpy as np
import pandas as pd
import joblib

from src.eval.shap_analysis import explain_for_g3

DATA = "data/processed"
CLF_PATH = os.path.join("models", "applications_fit_classifier.pkl")
OUT = os.path.join("reports", "shap", "per_prediction_examples.json")


def engineer(df):
    d = df.copy()
    cand = d["skills"].fillna("").str.split("|")
    req = d["required_skills"].fillna("").str.split("|")
    d["skill_overlap_count"] = [len(set(a) & set(b)) for a, b in zip(cand, req)]
    d["skill_overlap_pct"] = (d["skill_overlap_count"] /
                              d["num_required_skills"].replace(0, np.nan)).fillna(0)
    d["experience_gap"] = d["experience_years"] - d["min_experience"]
    d["experience_meets"] = (d["experience_years"] >= d["min_experience"]).astype(int)
    d["project_count"] = d["projects"].fillna("").str.split("|").apply(
        lambda x: len([p for p in x if p.strip()]))
    edu = d["education"].fillna("").str.lower()
    d["education_level"] = np.select(
        [edu.str.contains("phd"),
         edu.str.contains("m.s|master|mba|m.tech"),
         edu.str.contains("b.s|bachelor|b.tech|b.e")], [3, 2, 1], default=0)
    dt = pd.to_datetime(d["application_date"], errors="coerce")
    d["app_month"] = dt.dt.month
    d["app_weekday"] = dt.dt.weekday
    return d


def main():
    clf = joblib.load(CLF_PATH)
    features = list(clf.feature_names_in_)

    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    tef = engineer(test)

    sample = tef.sample(min(5, len(tef)), random_state=0)
    bundle = []
    for idx in sample.index:
        x = tef.loc[[idx], features]
        bundle.append({
            "candidate_id": int(test.loc[idx, "candidate_id"]),
            "job_id": int(test.loc[idx, "job_id"]),
            **explain_for_g3(clf, x, top_k=3),
            "model_version": "applications_fit_classifier_v1",
            "used_fallback": False,
        })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(bundle, f, indent=2)

    print(f"wrote {len(bundle)} per-prediction explanations (G3 schema) -> {OUT}\n")
    print("sample (one prediction):")
    print(json.dumps(bundle[0], indent=2))


if __name__ == "__main__":
    main()

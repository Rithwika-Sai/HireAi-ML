"""
M3 — SHAP explainability  (HireAI spec Section 8)

  * Week 3:  global feature importance across the test set.
  * Week 4:  PER-PREDICTION SHAP for M4's /score API and GenAI G3.

G3 explanation contract (agreed format) — `explain_prediction` returns a list of:
    {"feature_name": str, "contribution": float, "direction": "increases" | "decreases"}
sorted by |contribution|, top-k. `contribution` is the signed SHAP value; `direction`
says whether the feature pushed the prediction up or down.

Works with plain tree models (XGBoost regressor, RandomForest classifier).
Requires:  pip install shap matplotlib
"""

import numpy as np


# ---------------------------------------------------------------------------
# Week 3 — global importance
# ---------------------------------------------------------------------------
def global_importance(model, X, max_samples=500):
    """Mean |SHAP value| per feature across (a sample of) X. Returns sorted dict."""
    import shap

    Xs = X.iloc[:max_samples] if hasattr(X, "iloc") else X[:max_samples]
    sv = shap.TreeExplainer(model).shap_values(Xs)
    arr = np.array(sv)
    if isinstance(sv, list):                      # list of per-class arrays
        mean_abs = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
    elif arr.ndim == 3:                           # (samples, features, classes)
        mean_abs = np.abs(arr).mean(axis=(0, 2))
    else:                                         # regression (samples, features)
        mean_abs = np.abs(arr).mean(axis=0)
    names = list(X.columns) if hasattr(X, "columns") else [f"f{i}" for i in range(len(mean_abs))]
    return dict(sorted(zip(names, mean_abs), key=lambda t: -t[1]))


# ---------------------------------------------------------------------------
# Week 4 — per-prediction SHAP (for the API / G3)
# ---------------------------------------------------------------------------
def explain_prediction(model, x_row, top_k=5):
    """Top-k signed feature contributions for ONE prediction, in G3's JSON format.

    x_row : single-row DataFrame with exactly the model's feature columns.
    """
    import shap

    sv = shap.TreeExplainer(model).shap_values(x_row)
    names = list(x_row.columns)
    arr = np.array(sv)

    if hasattr(model, "classes_"):                # classifier -> use predicted class
        classes = list(model.classes_)
        ci = classes.index(model.predict(x_row)[0])
        if isinstance(sv, list):
            vals = np.ravel(np.array(sv[ci])[0])
        elif arr.ndim == 3:                       # (1, features, classes)
            vals = arr[0, :, ci]
        else:
            vals = arr[0]
    else:                                         # regressor
        vals = arr[0] if arr.ndim == 2 else np.ravel(arr)

    pairs = [
        {"feature_name": n,
         "contribution": round(float(v), 4),
         "direction": "increases" if v >= 0 else "decreases"}
        for n, v in zip(names, vals)
    ]
    pairs.sort(key=lambda p: abs(p["contribution"]), reverse=True)
    return pairs[:top_k]


def explain_for_g3(clf, x_row, top_k=3):
    """Per-prediction explanation in GenAI G3's agreed schema (classifier).

    Returns predicted_label, class probabilities, base_value, and the top-k
    positive / negative drivers — each with the signed `shap_value` AND the raw
    `feature_value` (so G3 can write "5 years of experience contributed +...").
    """
    import shap

    classes = list(clf.classes_)
    proba = clf.predict_proba(x_row)[0]
    pred = clf.predict(x_row)[0]
    ci = classes.index(pred)

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(x_row)
    arr = np.array(sv)
    if isinstance(sv, list):
        vals = np.ravel(np.array(sv[ci])[0])
    elif arr.ndim == 3:
        vals = arr[0, :, ci]
    else:
        vals = arr[0]

    ev = explainer.expected_value
    base = (float(np.ravel(ev)[ci]) if isinstance(ev, (list, np.ndarray))
            else float(ev) if ev is not None else None)

    names = list(x_row.columns)
    raw = x_row.iloc[0]
    feats = [{"feature_name": n,
              "shap_value": round(float(v), 4),
              "feature_value": round(float(raw[n]), 4)}
             for n, v in zip(names, vals)]
    pos = sorted([f for f in feats if f["shap_value"] > 0],
                 key=lambda f: -f["shap_value"])[:top_k]
    neg = sorted([f for f in feats if f["shap_value"] < 0],
                 key=lambda f: f["shap_value"])[:top_k]

    return {
        "predicted_label": str(pred),
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, proba)},
        "base_value": round(base, 4) if base is not None else None,
        "top_positive_drivers": pos,
        "top_negative_drivers": neg,
    }

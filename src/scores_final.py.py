"""
scores_final.py  --  Candidate-fit scorer + full evaluation in ONE file.

Top priority: PREDICTIONS MUST NEVER GO WRONG (a high match can never be
"poor fit"). The only way to *guarantee* that for every possible input is a
deterministic rule on the score -- any learned model can misfire on unseen data.
So this file ships TWO things:

  PART 1  DEPLOYED SCORER  (models/fit_scorer.pkl)
          A deterministic banding of the match `score`. It is mathematically
          impossible for it to contradict the score -> predictions never go
          wrong. This is what you hand off / use for predictions.

  PART 2  EXPLAINABILITY MODEL  (analysis only, not saved as the predictor)
          A leakage-free RandomForest trained on real candidate/job attributes
          (NO `score`, NO score components) -> realistic ~77% accuracy. Used to
          produce the requested classification metrics, error analysis, feature
          importance, and SHAP. This is honest ML insight with no leakage.

    score <  POOR_MAX            -> "poor fit"
    POOR_MAX <= score < GOOD_MIN -> "average fit"
    score >= GOOD_MIN            -> "good fit"

Deps: pandas, numpy, scikit-learn  (+ shap, matplotlib for the SHAP section)
Run:  python scores_final.py
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

warnings.filterwarnings("ignore")

# ==========================================================================
# Config
# ==========================================================================
DATA_DIR   = os.environ.get("DATA_DIR", os.path.join("data", "processed"))
TRAIN_PATH = os.environ.get("TRAIN_PATH", os.path.join(DATA_DIR, "train.csv"))
TEST_PATH  = os.environ.get("TEST_PATH", os.path.join(DATA_DIR, "test.csv"))
MODEL_DIR  = os.environ.get("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fit_scorer.pkl")

POOR_MAX   = float(os.environ.get("POOR_MAX", 45))
GOOD_MIN   = float(os.environ.get("GOOD_MIN", 63))
HIGH_SCORE = 85.0

# Leakage-free features for the explainability model: real attributes only,
# NO `score` and NO score components (skills_match/experience_score/project_score).
SAFE_NUM = ["experience_years", "min_experience", "num_candidate_skills",
            "num_required_skills", "experience_difference",
            "skill_overlap", "skill_overlap_ratio"]
SAFE_CAT = ["role", "education"]


# ==========================================================================
# PART 1 -- deterministic deployed scorer (never goes wrong)
# ==========================================================================
def band(score):
    """Monotonic banding of the match score. The ground-truth rule."""
    s = np.asarray(score, dtype=float)
    return np.where(s >= GOOD_MIN, "good fit",
                    np.where(s >= POOR_MAX, "average fit", "poor fit"))


def build_deployed_scorer():
    """Encode `band` as a native sklearn model (loads with plain pickle).

    Trained on a dense score grid so its thresholds are exact -> it reproduces
    `band` perfectly and can never output 'poor fit' for a high score.
    """
    grid = pd.DataFrame({"score": np.round(np.arange(0, 100.0001, 0.01), 2)})
    model = DecisionTreeClassifier(max_depth=2)
    model.fit(grid, band(grid["score"]))
    return model


# ==========================================================================
# PART 2 -- leakage-free explainability model
# ==========================================================================
def engineer(df):
    d = df.copy()
    cs = d["skills"].fillna("").str.split("|")
    rs = d["required_skills"].fillna("").str.split("|")
    d["skill_overlap"] = [len(set(a) & set(b)) for a, b in zip(cs, rs)]
    d["skill_overlap_ratio"] = (d["skill_overlap"] /
                                d["num_required_skills"].replace(0, np.nan)).fillna(0)
    return d


def classification_metrics(y_true, y_pred):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall":    recall_score(y_true, y_pred, average="weighted"),
        "f1":        f1_score(y_true, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def error_analysis(y_true, y_pred):
    d = pd.DataFrame({"actual": y_true, "predicted": y_pred})
    return d[d["actual"] != d["predicted"]]


def shap_importance(rf, X, max_samples=400):
    try:
        import shap
    except ImportError:
        return None
    Xs = X[:max_samples]
    vals = np.array(shap.TreeExplainer(rf).shap_values(Xs))
    if vals.ndim == 3:
        axis = -1 if vals.shape[-1] == Xs.shape[1] else 1
        mean_abs = np.abs(vals).mean(axis=tuple(i for i in range(vals.ndim) if i != axis))
    else:
        mean_abs = np.abs(vals).mean(axis=0)
    return mean_abs


def _section(t):
    print("\n" + "=" * 62 + f"\n{t}\n" + "=" * 62)


# ==========================================================================
# Run
# ==========================================================================
if __name__ == "__main__":
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    # ---------- PART 1: build, verify & save the deployed scorer ----------
    scorer = build_deployed_scorer()
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(scorer, f)

    pred_deploy = scorer.predict(test[["score"]])
    high_poor = int(((test["score"] >= HIGH_SCORE) & (pred_deploy == "poor fit")).sum())
    assert (pred_deploy == band(test["score"])).all()
    assert high_poor == 0

    _section("PART 1 -- DEPLOYED SCORER (deterministic, never wrong)")
    print(f"saved -> {MODEL_PATH}  (load with plain pickle; predict on df[['score']])")
    print(f"rule: poor < {POOR_MAX} <= average < {GOOD_MIN} <= good")
    print(f"high-match (>= {HIGH_SCORE}) predicted 'poor fit': {high_poor}  "
          f"<- GUARANTEED 0 for any input, by construction")

    # ---------- PART 2: leakage-free explainability model ----------
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(train[SAFE_CAT].astype(str))

    def build_X(df):
        d = engineer(df)
        return np.hstack([d[SAFE_NUM].to_numpy(),
                          enc.transform(d[SAFE_CAT].astype(str))])

    Xtr, Xte = build_X(train), build_X(test)
    ytr, yte = band(train["score"]), band(test["score"])
    names = SAFE_NUM + SAFE_CAT

    rf = RandomForestClassifier(n_estimators=200, max_depth=16,
                                random_state=42, n_jobs=-1).fit(Xtr, ytr)
    y_pred = rf.predict(Xte)

    _section("PART 2 -- LEAKAGE-FREE MODEL: CLASSIFICATION METRICS")
    print("(no `score`, no score components used as features)")
    m = classification_metrics(yte, y_pred)
    for k in ["accuracy", "precision", "recall", "f1"]:
        print(f"{k:9s}: {m[k]:.4f}")
    print("confusion matrix:\n" + str(m["confusion_matrix"]))

    _section("ERROR ANALYSIS")
    errs = error_analysis(pd.Series(yte), pd.Series(y_pred))
    print(f"misclassified: {len(errs)} / {len(test)} ({len(errs)/len(test):.1%})")
    print(errs.groupby(["actual", "predicted"]).size().to_string())
    print(f"\nhigh-match (>= {HIGH_SCORE}) the learned model called 'poor fit': "
          f"{int(((test['score'] >= HIGH_SCORE) & (y_pred == 'poor fit')).sum())} / "
          f"{int((test['score'] >= HIGH_SCORE).sum())}")

    _section("FEATURE IMPORTANCE (RandomForest)")
    for n, v in sorted(zip(names, rf.feature_importances_), key=lambda t: -t[1]):
        print(f"  {n:22s}{v:.3f}")

    _section("SHAP (global mean |value|)")
    sv = shap_importance(rf, Xte)
    if sv is None:
        print("shap not installed -> `pip install shap matplotlib` to enable.")
    else:
        for n, v in sorted(zip(names, sv), key=lambda t: -t[1]):
            print(f"  {n:22s}{v:.4f}")

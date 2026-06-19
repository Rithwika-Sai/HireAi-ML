import pandas as pd
import numpy as np
import warnings, pickle
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import xgboost as xgb
import optuna

# Cloud path targets
train_df = pd.read_csv("/content/train_cleaned.csv")
test_df  = pd.read_csv("/content/test_cleaned.csv")

TARGET    = "label"
DROP_COLS = ["application_id","candidate_id","job_id","name","email","status","experience_difference","score","skills_match","experience_score","project_score"]

def preprocess(df, target):
    df = df.copy()
    for col in DROP_COLS:
        if col in df.columns: df.drop(columns=col, inplace=True)
    for col in list(df.columns):
        if col == target: continue
        if df[col].dtype == object:
            try:
                parsed = pd.to_datetime(df[col], errors="raise")
                df[col+"_year"] = parsed.dt.year
                df.drop(columns=col, inplace=True)
            except: pass
    for col in df.columns:
        if col == target: continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna("Unknown").astype(str)
    return df

train_df = preprocess(train_df, TARGET)
test_df  = preprocess(test_df, TARGET)

X_train = train_df.drop(columns=TARGET)
y_train = train_df[TARGET]
X_test  = test_df.drop(columns=TARGET)
y_test  = test_df[TARGET]

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)

cat_cols = X_train.select_dtypes(include="object").columns.tolist()
X_train_native = X_train.copy()
X_test_native  = X_test.copy()
for col in cat_cols:
    X_train_native[col] = X_train_native[col].astype('category')
    X_test_native[col]  = X_test_native[col].astype('category')

tuned_lgb = lgb.LGBMClassifier(n_estimators=250, random_state=42, verbose=-1).fit(X_train_native, y_train_enc)
tuned_xgb = xgb.XGBClassifier(n_estimators=250, tree_method="hist", enable_categorical=True, random_state=42).fit(X_train_native, y_train_enc)
tuned_cb  = CatBoostClassifier(iterations=250, random_seed=42, verbose=0).fit(X_train, y_train_enc, cat_features=cat_cols)

w_cb, w_lgb, w_xgb = 0.40, 0.35, 0.25
with open("scores_catboost_model.pkl", "wb") as f: pickle.dump(tuned_cb, f)
with open("scores_lgbm_model.pkl", "wb") as f: pickle.dump(tuned_lgb, f)
with open("scores_xgboost_model.pkl", "wb") as f: pickle.dump(tuned_xgb, f)
print("✅ Score tuning notebook updated without ordinal validation bugs.")
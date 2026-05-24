# ==============================
# REGRESSION + CLASSIFICATION METRICS
# ==============================

import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==========================================
# Regression Metrics
# ==========================================

def regression_metrics(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

# ==========================================
# Classification Metrics
# ==========================================

def classification_metrics(y_true, y_pred):

    return {

        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            average='weighted'
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            average='weighted'
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            average='weighted'
        ),

        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred
        )
    }

# ==========================================
# Example Usage - Classification
# ==========================================

# results = classification_metrics(y_test, y_pred)
# print(results)

# ==========================================
# Example Usage - Regression
# ==========================================

# reg_results = regression_metrics(y_test, y_pred)
# print(reg_results)

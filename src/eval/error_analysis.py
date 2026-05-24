import pandas as pd

def classification_error_analysis(y_true, y_pred):

    error_df = pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred
    })

    errors = error_df[
        error_df["actual"] != error_df["predicted"]
    ]

    return errors

def regression_error_analysis(y_true, y_pred):

    error_df = pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred
    })

    error_df["error"] = (
        error_df["actual"] -
        error_df["predicted"]
    )

    return error_df
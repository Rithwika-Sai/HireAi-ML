import pandas as pd
def analyze_errors(df):
    # Wrong predictions
    wrong_predictions = df[df["actual"] != df["predicted"]]
    print("Total Errors:", len(wrong_predictions))
    return wrong_predictions
# Dummy sample data
data = {
    "actual": [1, 0, 1, 1, 0],
    "predicted": [1, 1, 1, 0, 0]
}
df = pd.DataFrame(data)
# Run analysis
print(analyze_errors(df))
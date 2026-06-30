import pandas as pd
import json

def load_dataset(file_path: str) -> str:
    df = pd.read_csv(file_path)
    return df.to_json(orient="records")

def analyze_dataset(file_path: str) -> str:
    df = pd.read_csv(file_path)

    analysis = {
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "numeric_summary": json.loads(df.describe().to_json()),
        "sample_rows": json.loads(df.head(3).to_json(orient="records"))
    }

    return json.dumps(analysis, indent=2)
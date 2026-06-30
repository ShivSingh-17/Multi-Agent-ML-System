import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

def is_text_column(series: pd.Series) -> bool:
    """
    Returns True if a column contains free-form text.
    Uses pd.api.types.is_string_dtype which handles both
    object dtype AND pandas StringDtype correctly.
    """
    if not pd.api.types.is_string_dtype(series):
        return False
    sample = series.dropna().head(200).astype(str)
    if len(sample) == 0:
        return False
    avg_len   = sample.str.len().mean()
    avg_words = sample.str.split().str.len().mean()
    return avg_len > 50 or avg_words > 8

def preprocess_dataset(file_path: str, target_column: str) -> str:
    df = pd.read_csv(file_path)

    # ── Step 1: Identify text columns BEFORE any dropping ──
    text_cols = [
        col for col in df.columns
        if col != target_column and is_text_column(df[col])
    ]

    # ── Step 2: Drop high-missing columns ──
    threshold = len(df) * 0.5
    df = df.dropna(thresh=threshold, axis=1)

    # ── Step 3: Drop non-informative columns — NEVER drop text cols ──
    drop_cols = [
        col for col in df.columns
        if col not in text_cols
        and col != target_column
        and (df[col].nunique() == len(df) or df[col].nunique() == 1)
    ]
    df = df.drop(columns=drop_cols, errors='ignore')

    # ── Step 4: Fill missing values ──
    for col in df.columns:
        if col in text_cols:
            df[col] = df[col].fillna('')
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else '')

    # ── Step 5: TF-IDF for text columns ──
    tfidf_info = {}
    for col in text_cols:
        if col not in df.columns:
            continue
        try:
            max_features = 300
            tfidf        = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2)
            )
            text_matrix = tfidf.fit_transform(df[col].astype(str)).toarray()
            text_df     = pd.DataFrame(
                text_matrix,
                columns=[f'tfidf_{i}' for i in range(text_matrix.shape[1])],
                index=df.index
            )
            df = df.drop(columns=[col])
            df = pd.concat([df, text_df], axis=1)
            tfidf_info[col] = text_matrix.shape[1]
        except Exception as e:
            print(f"  ⚠️  TF-IDF failed for '{col}': {e}")
            df = df.drop(columns=[col], errors='ignore')

    # ── Step 6: Label encode remaining categoricals ──
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        if col != target_column:
            df[col] = le.fit_transform(df[col].astype(str))

    if df[target_column].dtype == 'object':
        df[target_column] = le.fit_transform(df[target_column].astype(str))

    result = {
        "processed_shape":         {"rows": df.shape[0], "columns": df.shape[1]},
        "final_columns":           list(df.columns[:20]),
        "target_column":           target_column,
        "text_columns_vectorized": tfidf_info,
        "processed_data":          df.to_json(orient='records')
    }

    return json.dumps(result, indent=2)
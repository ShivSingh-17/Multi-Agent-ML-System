

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from state.ml_state import MLState
from tools.feature_tools import preprocess_dataset, is_text_column
import pandas as pd
import json
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

def feature_engineer_agent(state: MLState) -> MLState:
    print("\n" + "="*60)
    print("⚙️   AGENT 2: FEATURE ENGINEER")
    print("="*60)

    df = pd.read_csv(state["file_path"])
    print(f"  📥 Input shape           : {df.shape[0]} rows × {df.shape[1]} columns")

    # Detect text columns
    text_cols = [
        col for col in df.columns
        if col != state["target_column"] and is_text_column(df[col])
    ]
    if text_cols:
        print(f"\n  📝 Text columns detected — TF-IDF will be applied:")
        for col in text_cols:
            sample    = df[col].dropna().head(100).astype(str)
            avg_words = sample.str.split().str.len().mean()
            avg_len   = sample.str.len().mean()
            print(f"      • {col:<15} avg length: {avg_len:.0f} chars, avg words: {avg_words:.1f} → TF-IDF (300 features)")
    else:
        print(f"\n  ℹ️  No free-form text columns detected")

    # High-missing drops
    threshold    = len(df) * 0.5
    high_missing = [c for c in df.columns if df[c].isnull().sum() > threshold]
    if high_missing:
        print(f"\n  🗑️  Dropping high-missing columns (>50% missing):")
        for col in high_missing:
            pct = round(df[col].isnull().sum() / len(df) * 100, 1)
            print(f"      • {col} ({pct}% missing)")

    # Non-informative drops — text cols are exempt
    non_info = [
        c for c in df.columns
        if c not in text_cols
        and c != state["target_column"]
        and (df[c].nunique() == len(df) or df[c].nunique() == 1)
    ]
    if non_info:
        print(f"\n  🗑️  Dropping non-informative columns:")
        for col in non_info:
            print(f"      • {col} (unique values: {df[col].nunique()})")

    # Missing value strategy
    needs_fill = [
        c for c in df.columns
        if df[c].isnull().sum() > 0 and c not in high_missing
    ]
    if needs_fill:
        print(f"\n  🔧 Missing value strategy:")
        for col in needs_fill:
            if col in text_cols:        strategy = "empty string"
            elif pd.api.types.is_numeric_dtype(df[col]): strategy = "median"
            else:                       strategy = "mode"
            print(f"      • {col:<15} → {strategy}")

    # Label encoding
    cat_cols = [
        c for c in df.select_dtypes(include='object').columns
        if c not in high_missing and c not in text_cols
        and c != state["target_column"]
    ]
    if cat_cols:
        print(f"\n  🔠 Label encoding categorical columns:")
        for col in cat_cols:
            print(f"      • {col}")

    print(f"\n  ⚙️  Running preprocessing...")
    preprocessing_result = preprocess_dataset(state["file_path"], state["target_column"])
    full_result          = json.loads(preprocessing_result)

    print(f"  📤 Output shape          : {full_result['processed_shape']['rows']} rows × {full_result['processed_shape']['columns']} columns")

    for col, n in full_result.get("text_columns_vectorized", {}).items():
        print(f"  📊 TF-IDF applied        : '{col}' → {n} features created")

    shown = full_result['final_columns'][:5]
    more  = f" + {full_result['processed_shape']['columns'] - 5} more..." \
            if full_result['processed_shape']['columns'] > 5 else ""
    print(f"  ✅ Final features        : {shown}{more}")
    print(f"  ✅ Feature engineering complete")

    summary = {
        "processed_shape":         full_result["processed_shape"],
        "final_columns":           full_result["final_columns"],
        "target_column":           full_result["target_column"],
        "text_columns_vectorized": full_result.get("text_columns_vectorized", {})
    }

    prompt = f"""
You are an expert feature engineer. Review the preprocessing steps taken.

Original Analysis: {state["analysis_result"]}
Preprocessing Summary: {json.dumps(summary, indent=2)}

Summarize:
1. Text columns detected and how TF-IDF was applied
2. Columns dropped and why
3. Missing value handling
4. Encoding applied
5. Final feature set ready for modelling
Keep it concise and technical.
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        **state,
        "preprocessing_result": preprocessing_result,
        "current_agent":        "feature_engineer"
    }

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from state.ml_state import MLState
from tools.data_tools import analyze_dataset
import pandas as pd
import json
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

def detect_problem_type(file_path: str, target_column: str) -> str:
    df = pd.read_csv(file_path)
    target = df[target_column]
    unique_ratio = target.nunique() / len(target)
    if target.dtype == "object" or target.nunique() <= 10 or unique_ratio < 0.05:
        return "classification"
    return "regression"

def data_analyst_agent(state: MLState) -> MLState:
    print("\n" + "="*60)
    print("🔍  AGENT 1: DATA ANALYST")
    print("="*60)

    df = pd.read_csv(state["file_path"])
    print(f"  📂 Dataset loaded        : {state['file_path']}")
    print(f"  📊 Shape                 : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  🎯 Target column         : {state['target_column']}")

    print(f"\n  📋 Columns found:")
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing = df[col].isnull().sum()
        missing_pct = round(missing / len(df) * 100, 1)
        flag = "  ⚠️  missing" if missing_pct > 0 else ""
        print(f"      • {col:<15} [{dtype:<8}]  missing: {missing_pct}%{flag}")

    problem_type = detect_problem_type(state["file_path"], state["target_column"])
    print(f"\n  🧠 Problem type detected : {problem_type.upper()}")
    print(f"  🔢 Target unique values  : {df[state['target_column']].nunique()}")

    print(f"\n  🤖 Sending to LLM for pattern analysis...")
    raw_analysis = analyze_dataset(state["file_path"])

    prompt = f"""
You are an expert data analyst. Analyze this dataset and provide key insights.

Problem Statement: {state["problem_statement"]}
Problem Type Detected: {problem_type}
Dataset Analysis: {raw_analysis}

Provide:
1. Key observations about the data
2. Important patterns or correlations
3. Which columns look most useful for predicting: {state["target_column"]}
4. Data quality issues found
5. Confirm if {problem_type} is the correct problem type based on the target variable
Keep it concise and technical.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"  ✅ Analysis complete")

    return {
        **state,
        "analysis_result": response.content,
        "problem_type": problem_type,
        "current_agent": "data_analyst"
    }
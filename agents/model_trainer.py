

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from state.ml_state import MLState
from tools.model_tools import train_and_select_model
import json
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

def model_trainer_agent(state: MLState) -> MLState:
    print("\n" + "="*60)
    print("🤖  AGENT 3: MODEL TRAINER")
    print("="*60)
    print(f"  🧠 Problem type          : {state['problem_type'].upper()}")
    print(f"  🎯 Target column         : {state['target_column']}")
    print(f"\n  🏋️  Training models and running 5-fold cross validation...")

    model_result = train_and_select_model(
        state["preprocessing_result"],
        state["target_column"]
    )

    model_data = json.loads(model_result)
    metric_label = model_data["best_metric_label"]

    print(f"\n  📊 Model comparison results:")
    print(f"  {'Model':<25} {'CV Score':>10} {'CV Std':>10} {'Test Score':>12}")
    print(f"  {'-'*57}")

    for name, res in model_data["all_results"].items():
        if "error" not in res:
            cv = res.get("cv_mean_score", 0)
            std = res.get("cv_std", 0)
            test = res.get("primary_metric", 0)
            marker = "  ⭐ BEST" if name == model_data["best_model"] else ""
            print(f"  {name:<25} {cv:>10.4f} {std:>10.4f} {test:>12.4f}{marker}")
        else:
            print(f"  {name:<25} {'ERROR':>10}")

    print(f"\n  🏆 Best model            : {model_data['best_model']}")
    print(f"  📈 Best {metric_label:<16} : {model_data['best_metric_value']}")

    model_summary = {
        "problem_type": model_data["problem_type"],
        "best_model": model_data["best_model"],
        "best_metric_label": metric_label,
        "best_metric_value": model_data["best_metric_value"],
        "all_results": {
            name: {k: v for k, v in res.items() if k != "error"}
            for name, res in model_data["all_results"].items()
            if "error" not in res
        }
    }

    prompt = f"""
You are an expert ML engineer. Analyse these model training results.

Model Results: {json.dumps(model_summary, indent=2)}

Provide:
1. Which model performed best and why
2. Comparison of all models tried
3. What the {metric_label} score means in context
4. Any concerns about overfitting or underfitting
Keep it concise and technical.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"  ✅ Model training complete")

    return {
        **state,
        "model_result": model_result,
        "current_agent": "model_trainer"
    }
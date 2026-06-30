from graph.workflow import build_workflow
from state.ml_state import MLState

def run_pipeline(file_path: str, problem_statement: str, target_column: str):

    initial_state: MLState = {
        "file_path": file_path,
        "problem_statement": problem_statement,
        "target_column": target_column,
        "analysis_result": None,
        "preprocessing_result": None,
        "model_result": None,
        "report": None,
        "current_agent": None,
        "error": None
    }

    print("🚀 Starting Multi-Agent ML Pipeline...\n")
    print(f"📂 Dataset     : {file_path}")
    print(f"🎯 Target      : {target_column}")
    print(f"📋 Problem     : {problem_statement}\n")
    print("=" * 60)

    app = build_workflow()
    final_state = app.invoke(initial_state)

    print("\n" + "=" * 60)
    print("\n✅ Pipeline Complete!\n")
    print("=" * 60)
    print("\n📄 FINAL REPORT\n")
    print(final_state["report"])

    return final_state

if __name__ == "__main__":
    run_pipeline(
        file_path="data/House Price Prediction Dataset.csv",
        problem_statement="Predict the price of house",
        target_column="Price"
    )
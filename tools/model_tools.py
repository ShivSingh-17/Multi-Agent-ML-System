import pandas as pd
import json
import io
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             mean_squared_error, mean_absolute_error, r2_score)
import numpy as np

def detect_problem_type(df: pd.DataFrame, target_column: str) -> str:
    target = df[target_column]
    unique_ratio = target.nunique() / len(target)

    if target.dtype == "object":
        return "classification"
    elif target.nunique() <= 10 or unique_ratio < 0.05:
        return "classification"
    else:
        return "regression"

def get_models(problem_type: str) -> dict:
    if problem_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(kernel="rbf", random_state=42)
        }
    else:
        return {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "KNN": KNeighborsRegressor(),
            "SVR": SVR(kernel="rbf")
        }

def evaluate_model(model, X_train, X_test, y_train, y_test, problem_type):
    cv_scoring = "accuracy" if problem_type == "classification" else "r2"
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=cv_scoring)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if problem_type == "classification":
        return {
            "cv_mean_score": round(cv_scores.mean(), 4),
            "cv_std": round(cv_scores.std(), 4),
            "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred, average="weighted"), 4),
            "primary_metric": round(accuracy_score(y_test, y_pred), 4)
        }
    else:
        return {
            "cv_mean_score": round(cv_scores.mean(), 4),
            "cv_std": round(cv_scores.std(), 4),
            "r2_score": round(r2_score(y_test, y_pred), 4),
            "mae": round(mean_absolute_error(y_test, y_pred), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            "primary_metric": round(r2_score(y_test, y_pred), 4)
        }

def train_and_select_model(processed_data_json: str, target_column: str) -> str:
    data = json.loads(processed_data_json)
    df = pd.read_json(io.StringIO(data["processed_data"]), orient="records")

    problem_type = detect_problem_type(df, target_column)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = get_models(problem_type)
    results = {}

    for name, model in models.items():
        try:
            results[name] = evaluate_model(
                model, X_train, X_test, y_train, y_test, problem_type
            )
        except Exception as e:
            results[name] = {"error": str(e), "primary_metric": -999}

    # Pick best by primary metric
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if not valid_results:
        raise ValueError(
            "All models failed to train. This usually means no feature columns "
            "remain after preprocessing. Check that your dataset has usable "
            "features and that the target column is correctly specified."
        )
    best_model = max(valid_results, key=lambda x: valid_results[x]["primary_metric"])
    metric_label = "accuracy" if problem_type == "classification" else "r2_score"

    return json.dumps({
        "problem_type": problem_type,
        "all_results": results,
        "best_model": best_model,
        "best_metric_label": metric_label,
        "best_metric_value": valid_results[best_model]["primary_metric"]
    }, indent=2)
from typing import TypedDict, Optional

class MLState(TypedDict):
    file_path: str
    problem_statement: str
    target_column: str
    problem_type: Optional[str]        # "classification" or "regression"
    analysis_result: Optional[str]
    preprocessing_result: Optional[str]
    model_result: Optional[str]
    report: Optional[str]
    current_agent: Optional[str]
    error: Optional[str]
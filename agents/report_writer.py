import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from state.ml_state import MLState

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

def truncate_text(text, max_chars=6000):
    text_str = str(text)
    if len(text_str) > max_chars:
        return text_str[:max_chars] + "\n...[TRUNCATED TO FIT API LIMITS]..."
    return text_str

def report_writer_agent(state: MLState) -> MLState:
    print("\n" + "="*60)
    print("🤖  AGENT 4: REPORT WRITER")
    print("="*60)
    print(f"  📝 Generating comprehensive ML report...")

    problem_stmt = state.get("problem_statement", "N/A")
    target = state.get("target_column", "N/A")
    
    # Aggressively truncate the data to ensure it stays well under the 12,000 token limit
    analysis = truncate_text(state.get("analysis_result", "N/A"), 6000)
    preprocessing = truncate_text(state.get("preprocessing_result", "N/A"), 6000)
    model_res = truncate_text(state.get("model_result", "N/A"), 8000)

    prompt = f"""
You are an expert Machine Learning Consultant. Your task is to write a final, comprehensive report based on the results of our auto-ML pipeline.
Some data may be truncated to fit limits, but do your best to summarize the key points shown.

Problem Statement: {problem_stmt}
Target Column: {target}

Data Analysis Summary:
{analysis}

Preprocessing Summary:
{preprocessing}

Model Training Results:
{model_res}

Write a professional, easy-to-understand markdown report. Include:
1. Executive Summary
2. Data Insights (key findings from analysis)
3. Preprocessing Steps (how data was cleaned/transformed)
4. Model Performance (which model won and what the metrics mean)
5. Conclusion & Recommendations
"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        report_content = response.content
    except Exception as e:
        print(f"  ❌ Error generating report: {e}")
        report_content = f"Failed to generate report due to API error: {e}"

    # Save to file
    with open("final_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"  ✅ Report generation complete. Saved as final_report.md")

    return {
        **state,
        "report": report_content,
        "current_agent": "report_writer"
    }
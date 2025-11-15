import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from datetime import datetime
import numpy as np
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from config import settings, BASE_DIR
import traceback
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

# Configure generative AI
genai.configure(api_key=settings.GOOGLE_API_KEY)
USER_HOME = os.path.expanduser("~")
OUTPUT_DIR = os.path.join(USER_HOME, "Desktop")
TEMP_CHART_DIR = os.path.join(OUTPUT_DIR, "jarvix_temp_charts")
os.makedirs(TEMP_CHART_DIR, exist_ok=True)

# Safety settings for Jarvix API calls to prevent unnecessary blocking.
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}


# --- DYNAMIC MULTI-STEP AI ANALYSIS ---
def get_ai_plan(df_head: str, columns: List[str]) -> Dict[str, Any]:
    """Step 1: Ask Jarvix to create a strategic plan as a JSON object."""
    model = genai.GenerativeModel("gemini-flash-latest")
    prompt = f"""
You are a strategic data analyst. Based on the data preview and columns, create an analysis plan.

**Data Preview (df.head()):**
{df_head}

**Columns available:**
{columns}

**Your output MUST be a valid JSON object** with the following keys:
1.  `"strategic_recommendations"`: A list of strings with advice (e.g., "Drop column X", "Create feature Y").
2.  `"feature_engineering_code"`: A string containing Python code to preprocess the data and create the new features you recommended. This code will modify the DataFrame `df`. IMPORTANT: Only use variables that are already defined (df, pd, np, os). Do NOT reference variables that don't exist. Each line should be self-contained and not depend on variables created in previous lines unless you explicitly create them.
3.  `"visualization_code"`: A string containing Python code to generate at least four insightful visualizations. This code MUST use the processed `df` and save each plot to a file in `TEMP_CHART_DIR`, appending the path to a `chart_paths` list. IMPORTANT: Use `plt.savefig()` to save each chart, then append the path to `chart_paths`, then call `plt.close()` to free memory.
4.  `"report_structure"`: A detailed structure for the PDF report with multiple sections. This should be a list of section objects, each with:
    - "title": The section title
    - "content": The main content for this section (as markdown) - **IMPORTANT: Each content section should be detailed and comprehensive, providing in-depth analysis and insights. Aim for at least 3-4 paragraphs of detailed explanation per section to ensure the PDF report fills multiple pages with valuable content.**
    - "type": The type of section ("text", "metrics", "recommendations", "conclusion", "charts")
    - "metrics" (optional): A dictionary of key metrics to display in this section (only for "metrics" type)

Create a comprehensive, multi-page report with 5-7 sections that tell a story about the data. Include:
- An introduction section explaining the dataset with detailed background information
- Multiple sections with key insights and metrics, each providing extensive analysis
- A section with visual analysis (charts) with detailed explanations of each visualization
- A conclusion with comprehensive recommendations
- Any other relevant sections with substantial content

**IMPORTANT: Each section should contain detailed, comprehensive content that fills multiple pages. Avoid brief or superficial descriptions. Each content section should be at least 3-4 paragraphs long with detailed analysis and insights.**

Example JSON output structure:
```json
{{
    "strategic_recommendations": [
        "The 'timestamp' column is critical for time-series analysis.",
        "The 'ip_address' can be used to approximate unique visitors."
    ],
    "feature_engineering_code": "df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')\\ndf.dropna(subset=['timestamp'], inplace=True)\\ndf['hour'] = df['timestamp'].dt.hour\\ndf['visits_per_ip'] = df.groupby('ip_address')['ip_address'].transform('count')",
    "visualization_code": "plt.figure(figsize=(12, 6))\\nsns.countplot(data=df, x='hour', palette='viridis')\\nplt.title('Visits by Hour')\\nchart_path = os.path.join(TEMP_CHART_DIR, 'hourly_visits.png')\\nplt.savefig(chart_path)\\nchart_paths.append(chart_path)\\nplt.close()\\n\\nplt.figure(figsize=(10, 5))\\nsns.histplot(df['visits_per_ip'], bins=20, kde=True, palette='viridis')\\nplt.title('Distribution of Visits Per IP')\\nchart_path = os.path.join(TEMP_CHART_DIR, 'visits_per_ip_distribution.png')\\nplt.savefig(chart_path)\\nchart_paths.append(chart_path)\\nplt.close()\\n\\nplt.figure(figsize=(14, 7))\\nsns.lineplot(data=df.groupby('hour').size().reset_index(name='count'), x='hour', y='count')\\nplt.title('Hourly Visit Trends')\\nchart_path = os.path.join(TEMP_CHART_DIR, 'hourly_trends.png')\\nplt.savefig(chart_path)\\nchart_paths.append(chart_path)\\nplt.close()\\n\\nplt.figure(figsize=(12, 6))\\nsns.boxplot(data=df, x='hour', y='visits_per_ip', palette='viridis')\\nplt.title('Visits Per IP by Hour')\\nchart_path = os.path.join(TEMP_CHART_DIR, 'visits_per_ip_by_hour.png')\\nplt.savefig(chart_path)\\nchart_paths.append(chart_path)\\nplt.close()",
    "report_structure": [
        {{
            "title": "Dataset Overview",
            "content": "This comprehensive report provides a detailed analysis of the provided dataset containing X rows and Y columns. The data includes extensive information about various aspects of the domain being studied.\\n\\nIn this section, we will explore the fundamental characteristics of the dataset, including its structure, data types, and overall quality. We will examine the completeness of the data, identify any potential issues or anomalies, and provide insights into the data collection process.\\n\\nFurthermore, we will discuss the context in which this data was collected, the potential sources of bias or limitations, and how these factors might impact our analysis. This foundational understanding is crucial for interpreting the subsequent findings and recommendations.\\n\\nBy thoroughly understanding the dataset's composition and characteristics, we can ensure that our analysis is both accurate and meaningful, leading to actionable insights that can drive informed decision-making.",
            "type": "text"
        }},
        {{
            "title": "Key Metrics",
            "content": "The following comprehensive metrics provide a detailed overview of the dataset characteristics and key performance indicators. These metrics have been carefully calculated and selected to offer meaningful insights into the underlying patterns and trends within the data.\\n\\nEach metric represents a specific aspect of the dataset that is crucial for understanding the overall landscape and identifying areas of interest or concern. The values presented here serve as a foundation for the more detailed analysis and recommendations that follow in subsequent sections of this report.\\n\\nBy examining these metrics in conjunction with the visual representations and narrative analysis, stakeholders can gain a holistic understanding of the dataset and make informed decisions based on the evidence presented.",
            "type": "metrics",
            "metrics": {{
                "total_records": 1000,
                "unique_values": 42
            }}
        }},
        {{
            "title": "Visual Analysis",
            "content": "The following comprehensive charts and visualizations provide detailed insights into the key patterns, trends, and relationships within the dataset. Each visualization has been carefully selected and designed to highlight specific aspects of the data that is crucial for understanding the overall landscape.\\n\\nThese visual representations complement the quantitative metrics and narrative analysis presented in other sections of this report, offering a multi-dimensional perspective on the dataset. By examining these charts in conjunction with the other findings, stakeholders can gain a deeper understanding of the underlying patterns and dynamics within the data.\\n\\nEach visualization is accompanied by detailed explanations and interpretations that highlight the key insights and implications. These visual analyses serve as a foundation for the recommendations and strategic insights presented in the conclusion of this report.",
            "type": "charts"
        }},
        {{
            "title": "Conclusion & Recommendations",
            "content": "Based on our comprehensive analysis of the dataset, we have identified several key insights and patterns that warrant careful consideration.\\n\\nOur findings reveal significant trends and relationships within the data that have important implications for decision-making and strategic planning. These insights are derived from rigorous statistical analysis and careful examination of the visual representations of the data.\\n\\nMoving forward, we recommend implementing a series of targeted actions designed to capitalize on the opportunities identified in our analysis while addressing any potential challenges or areas of concern. These recommendations are grounded in the evidence presented throughout this report and are intended to maximize the value derived from the dataset.\\n\\nIt is important to note that the success of these recommendations will depend on careful implementation and ongoing monitoring of key performance indicators. We suggest establishing a framework for regular review and adjustment of these strategies based on evolving data and changing circumstances.",
            "type": "conclusion"
        }}
    ]
}}
```

**CRITICAL RULES:**
- In feature_engineering_code: Only use variables df, pd, np, os. Create new columns directly on df (e.g., df['new_col'] = ...). Do NOT create intermediate variables unless you use them immediately in the same line.
- In visualization_code: Only use variables df, pd, np, plt, sns, os, TEMP_CHART_DIR, chart_paths. Do NOT reference variables from feature_engineering_code unless they are columns in df.
- Each line of code should be independent and not rely on variables created in previous lines (unless they are DataFrame columns).
- For the report_structure, create a compelling narrative that guides the reader through the analysis.
- In feature_engineering_code: Do NOT create intermediate variables that are not immediately used to create a new column in the DataFrame. All operations should directly modify `df` or create new columns on `df`.
"""
    response = None
    try:
        response = model.generate_content(prompt, safety_settings=SAFETY_SETTINGS)
        # Clean the response to ensure it's valid JSON
        json_text = response.text

        # Attempt to extract JSON from a markdown code block first
        json_match = re.search(r"```json\n([\s\S]*?)```", json_text)
        if json_match:
            json_text = json_match.group(1)

        try:
            plan = json.loads(json_text)
        except json.JSONDecodeError:
            # If direct parsing fails, try to find a JSON object within the text
            json_match = re.search(r"\{[\s\S]*\}", json_text)
            if json_match:
                try:
                    plan = json.loads(json_match.group(0))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Failed to parse AI plan. Could not decode JSON from response. Error: {e}\nRaw response: {response.text}"
                    )
            else:
                raise ValueError(
                    f"Failed to parse AI plan. No JSON object found in response.\nRaw response: {response.text}"
                )
        return plan
    except (ValueError, AttributeError) as e:
        raise ValueError(
            f"Failed to parse AI plan. Jarvix's raw response might be invalid. Error: {e}\nResponse: {response.text}"
        )


# --- REPORTING ---
def generate_pdf_report(
    base_dir: Path,
    csv_path: str,
    summary: Dict[str, Any],
    charts: List[str],
    ai_report_structure: Optional[List[Dict]] = None,
) -> str:
    """Generates a PDF report from the analysis results."""
    templates_dir = base_dir / "templates_pdf"
    env = Environment(loader=FileSystemLoader(templates_dir))

    try:
        template = env.get_template("professional_report_template.html")
    except Exception as e:
        raise FileNotFoundError(
            f"Could not find 'professional_report_template.html' in '{templates_dir}'. Error: {e}"
        )

    now = datetime.now()
    base_path = Path(csv_path)
    dataset_slug = base_path.stem
    dataset_label = dataset_slug.replace("_", " ").replace("-", " ").title()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # If we have AI-generated report structure, use it; otherwise use the default approach
    if ai_report_structure:
        # Process the AI-generated report structure
        processed_sections = []
        chart_index = 0

        for section in ai_report_structure:
            processed_section = {
                "title": section.get("title", ""),
                "content": section.get("content", ""),
                "type": section.get("type", "text"),
                "metrics": section.get("metrics", {}),
            }

            # If this is a section that should include charts, add them
            if section.get("type") == "charts" and chart_index < len(charts):
                processed_section["chart_paths"] = charts[
                    chart_index : chart_index + 2
                ]  # Add up to 2 charts per section
                chart_index += 2

            processed_sections.append(processed_section)

        # Add any remaining charts to a final charts section
        if chart_index < len(charts):
            processed_sections.append(
                {
                    "title": "Additional Visualizations",
                    "content": "The following additional charts provide further insights into the data.",
                    "type": "charts",
                    "chart_paths": charts[chart_index:],
                }
            )

        template_vars = {
            "report_title": f"Automated Analysis of {base_path.name}",
            "custom_title": f"Jarvix Insights · {dataset_label}",
            "generation_date": now.strftime("%Y-%m-%d %H:%M:%S"),
            "file_name": base_path.name,
            "file_path": csv_path,
            "sections": processed_sections,
            "chart_pairs": [
                charts[i : i + 2] for i in range(0, len(charts), 2)
            ],  # Group charts for AI structure
        }
    else:
        # Fallback to the original approach
        # Group charts into pairs for PDF layout
        chart_pairs = [charts[i : i + 2] for i in range(0, len(charts), 2)]

        template_vars = {
            "report_title": f"Automated Analysis of {base_path.name}",
            "custom_title": f"Jarvix Insights · {dataset_label}",
            "generation_date": now.strftime("%Y-%m-%d %H:%M:%S"),
            "file_name": base_path.name,
            "file_path": csv_path,
            "strategic_recommendations": summary.pop("strategic_recommendations", []),
            "analysis_summary": summary,
            "chart_pairs": chart_pairs,  # Pass grouped charts
        }

    html_out = template.render(template_vars)
    report_filename = f"Jarvix_Report_{dataset_slug}_{timestamp}.pdf"
    report_path = os.path.join(OUTPUT_DIR, report_filename)

    try:
        HTML(string=html_out, base_url=str(TEMP_CHART_DIR)).write_pdf(report_path)
    except Exception as e:
        raise IOError(f"Failed to generate PDF report. Error: {e}")

    return report_path


# --- MAIN PIPELINE ---
def run_dynamic_analysis(base_dir: Path, file_path: str) -> str:
    """The main multi-step function Jarvix will call."""
    try:
        # Ensure TEMP_CHART_DIR exists before starting analysis
        os.makedirs(TEMP_CHART_DIR, exist_ok=True)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at '{file_path}'.")
        df = (
            pd.read_csv(file_path)
            if file_path.endswith(".csv")
            else pd.read_excel(file_path)
        )

        # --- STEP 1: GET THE AI'S STRATEGIC PLAN ---
        ai_plan = get_ai_plan(df.head().to_string(), list(df.columns))
        feature_engineering_code = ai_plan.get("feature_engineering_code", "")
        visualization_code = ai_plan.get("visualization_code", "")
        strategic_recommendations = ai_plan.get("strategic_recommendations", [])
        report_structure = ai_plan.get("report_structure", None)

        # --- STEP 2: EXECUTE FEATURE ENGINEERING ---
        # Create a safe savefig wrapper that ensures directories exist
        original_savefig = plt.savefig

        def safe_savefig(*args, **kwargs):
            """Wrapper for plt.savefig that ensures parent directory exists."""
            if args:
                filepath = args[0]
                parent_dir = os.path.dirname(filepath)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)
            return original_savefig(*args, **kwargs)

        # Monkey-patch plt.savefig to use our safe version
        plt.savefig = safe_savefig

        local_scope = {
            "df": df,
            "pd": pd,
            "plt": plt,
            "sns": sns,
            "np": np,
            "os": os,
            "TEMP_CHART_DIR": TEMP_CHART_DIR,
            "chart_paths": [],
            "analysis_summary": {
                "strategic_recommendations": strategic_recommendations
            },
        }
        if feature_engineering_code:
            try:
                exec(feature_engineering_code, globals(), local_scope)
            except NameError as e:
                error_msg = f"NameError in feature engineering code: {str(e)}. The code tried to use a variable that doesn't exist."
                return f"❌ **Feature Engineering Error:** {error_msg}\n\nPlease ensure the code only uses variables that are already defined (df, pd, np, os)."
            except Exception as e:
                error_msg = f"Error in feature engineering code: {str(e)}"
                return f"❌ **Feature Engineering Error:** {error_msg}"

        # --- STEP 3: EXECUTE VISUALIZATIONS ---
        if visualization_code:
            # Ensure chart_paths list exists in local scope
            if "chart_paths" not in local_scope:
                local_scope["chart_paths"] = []
            try:
                exec(visualization_code, globals(), local_scope)
            except NameError as e:
                error_msg = f"NameError in visualization code: {str(e)}. The code tried to use a variable that doesn't exist."
                return f"❌ **Visualization Error:** {error_msg}\n\nPlease ensure the code only uses variables that are already defined (df, pd, np, plt, sns, os, TEMP_CHART_DIR, chart_paths)."
            except Exception as e:
                error_msg = f"Error in visualization code: {str(e)}"
                return f"❌ **Visualization Error:** {error_msg}"

        # Restore original plt.savefig
        plt.savefig = original_savefig

        chart_paths = local_scope.get("chart_paths", [])
        analysis_summary = local_scope.get("analysis_summary", {})
        if not chart_paths:
            return "⚠️ **Warning:** The AI-generated plan did not produce any valid chart files."

        # --- STEP 4: GENERATE PDF ---
        report_path = generate_pdf_report(
            base_dir, file_path, analysis_summary, chart_paths, report_structure
        )

        # Clean up temporary chart images
        for path in chart_paths:
            if os.path.exists(path):
                os.remove(path)

        return f"👍 **Success:** AI-driven analysis is complete. Strategic report saved to: `{report_path}`"
    except Exception:
        tb_lines = traceback.format_exc().splitlines()
        error_details = "\n".join(tb_lines)
        return f"❌ **Data Science Agent Error:** A critical error occurred.\n<pre><code>{error_details}</code></pre>"

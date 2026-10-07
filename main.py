import os
import io
import ast
import json
import math
import pickle
import zipfile
import tempfile
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from dotenv import load_dotenv


# =========================================================
# PROJECT SETUP
# =========================================================

app = FastAPI(title="Trustworthy Data Analyst")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Add it to the .env file."
    )

client = genai.Client(api_key=API_KEY)

# Current model suggested by your Gemini API error
MODEL_NAME = "gemini-3.8-flash"


# =========================================================
# GLOBAL DATA STORAGE
# =========================================================

current_tables = {}
current_df = None


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():
    return {
        "message": "Trustworthy Data Analyst API is running!",
        "model": MODEL_NAME
    }


# =========================================================
# LOAD DATASET
# =========================================================

def load_csv(data, name):
    return pd.read_csv(io.BytesIO(data))


def load_excel(data):
    """
    Load all Excel sheets.
    Each sheet becomes a separate table.
    """
    sheets = pd.read_excel(
        io.BytesIO(data),
        sheet_name=None
    )

    return sheets


def make_table_name(name):
    """
    Create a clean table name from a filename.
    """
    name = Path(name).stem
    name = name.replace(" ", "_")
    name = name.replace("-", "_")
    return name


@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):

    global current_tables
    global current_df

    contents = await file.read()

    filename = file.filename.lower()

    tables = {}

    # =====================================================
    # CSV
    # =====================================================

    if filename.endswith(".csv"):

        df = load_csv(contents, file.filename)

        table_name = make_table_name(file.filename)

        tables[table_name] = df

    # =====================================================
    # EXCEL
    # =====================================================

    elif filename.endswith(".xlsx"):

        sheets = load_excel(contents)

        for sheet_name, df in sheets.items():

            table_name = str(sheet_name)

            tables[table_name] = df

    # =====================================================
    # ZIP
    # =====================================================

    elif filename.endswith(".zip"):

        try:

            with zipfile.ZipFile(io.BytesIO(contents)) as z:

                for name in z.namelist():

                    if name.endswith("/"):
                        continue

                    lower_name = name.lower()

                    if lower_name.endswith(".csv"):

                        data = z.read(name)

                        table_name = make_table_name(name)

                        tables[table_name] = pd.read_csv(
                            io.BytesIO(data)
                        )

                    elif lower_name.endswith(".xlsx"):

                        data = z.read(name)

                        sheets = pd.read_excel(
                            io.BytesIO(data),
                            sheet_name=None
                        )

                        base_name = make_table_name(name)

                        for sheet_name, df in sheets.items():

                            table_name = f"{base_name}_{sheet_name}"

                            tables[table_name] = df

        except zipfile.BadZipFile:

            return {
                "success": False,
                "status": "INVALID_FILE",
                "reason": "The uploaded ZIP file is invalid or corrupted."
            }

    else:

        return {
            "success": False,
            "status": "UNSUPPORTED_FILE",
            "reason": "Only CSV, XLSX and ZIP files are supported."
        }

    # =====================================================
    # CHECK DATA
    # =====================================================

    if not tables:

        return {
            "success": False,
            "status": "NO_DATA",
            "reason": "No CSV or Excel data was found."
        }

    # Save globally
    current_tables = tables

    # First table becomes df
    first_table_name = list(tables.keys())[0]
    current_df = tables[first_table_name]

    # =====================================================
    # DATA QUALITY REPORT
    # =====================================================

    table_information = {}

    for table_name, df in tables.items():

        missing_values = {
            str(column): int(count)
            for column, count in df.isnull().sum().items()
            if count > 0
        }

        table_information[table_name] = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": list(df.columns),
            "data_types": {
                str(column): str(dtype)
                for column, dtype in df.dtypes.items()
            },
            "missing_values": missing_values,
            "duplicate_rows": int(df.duplicated().sum()),
            "numeric_columns": list(
                df.select_dtypes(include="number").columns
            )
        }

    return {
        "success": True,
        "status": "DATASET_LOADED",
        "filename": file.filename,
        "tables_loaded": list(tables.keys()),
        "main_table": first_table_name,
        "table_information": table_information
    }


# =========================================================
# DATA CONTEXT FOR GEMINI
# =========================================================

def build_data_context():

    context = ""

    for table_name, df in current_tables.items():

        context += "\n"
        context += "=" * 60
        context += f"\nTABLE: {table_name}\n"
        context += "=" * 60

        context += f"\nRows: {len(df)}"
        context += f"\nColumns: {len(df.columns)}"

        context += "\nColumns and Data Types:\n"

        for column in df.columns:

            context += (
                f"- {column}: "
                f"{df[column].dtype}\n"
            )

        missing = df.isnull().sum()

        context += "\nMissing Values:\n"

        for column, count in missing.items():

            if count > 0:

                context += (
                    f"- {column}: {int(count)}\n"
                )

        duplicate_count = int(df.duplicated().sum())

        context += (
            f"\nDuplicate Rows: {duplicate_count}\n"
        )

        context += "\nSample Rows:\n"

        sample = df.head(5)

        context += sample.to_string(index=False)

        context += "\n"

    return context


# =========================================================
# AI PROMPT
# =========================================================

def build_ai_prompt(question):

    data_context = build_data_context()

    prompt = f"""
You are the reasoning engine of a trustworthy AI data analyst.

The user asks a natural-language question about uploaded datasets.

IMPORTANT:
Do NOT answer the question yourself.

Your job is to determine whether the question can be answered reliably
and, if it can, generate Python/Pandas code that will calculate the
answer from the actual dataset.

The backend will EXECUTE your generated code.

Therefore:
- Never invent numbers.
- Never calculate the final number mentally.
- Never assume a column exists.
- Never assume Sales means Revenue.
- Never assume missing information.
- Never ignore contradictions.
- Never silently guess.
- Every numerical answer must come from executed code.

AVAILABLE PYTHON OBJECTS:

df
    The first/main dataframe.

tables
    Dictionary containing all uploaded tables.

pd
    Pandas.

np
    NumPy.

Do NOT import any modules.

------------------------------------------------------------
TRUSTWORTHINESS RULES
------------------------------------------------------------

1. Check whether the requested information actually exists.

2. If the question asks for Revenue but only Sales exists,
   do NOT automatically treat Sales as Revenue.

3. If the question asks for production cost and no production-cost
   information exists, refuse.

4. If the question asks for growth/trend over time, check whether
   a usable date/year/time column exists.

5. If a question is ambiguous, refuse rather than guessing.

Example:
"Which product is best?"
This is ambiguous unless "best" has a measurable criterion.

6. If required columns are missing, refuse.

7. If data contains serious missing values relevant to the question,
   consider whether the result can still be calculated reliably.

8. If duplicate records can change the requested result,
   mention this and be cautious.

9. If multiple tables are required, use the appropriate tables
   and their relationships.

10. If units are inconsistent or unknown, refuse.

11. If the question cannot be answered reliably from the data,
    return CANNOT_DETERMINE.

12. Never fabricate a relationship between tables.

------------------------------------------------------------
SUPPORTED MATHEMATICAL ANALYSIS
------------------------------------------------------------

The system must automatically understand mathematical and analytical
questions from natural language. Do NOT hard-code question keywords.

When the required columns exist, generate executable Pandas code for:

1. TOTAL / SUM
Example:
"What is the total sales?"
result = df["Sales"].sum()

2. AVERAGE / MEAN
Example:
"What is the average profit?"
result = df["Profit"].mean()

3. MINIMUM / MAXIMUM
Use min() or max() on the requested numerical column.

4. DIFFERENCE
Example:
"What is the difference between total sales and total profit?"
result = df["Sales"].sum() - df["Profit"].sum()

5. RATIO
Example:
"What is the ratio of profit to sales?"
result = df["Profit"].sum() / df["Sales"].sum()
Only calculate when both columns exist and the denominator is non-zero.

6. PROFIT MARGIN
Example:
"What is the profit margin?"
result = (df["Profit"].sum() / df["Sales"].sum()) * 100
Only calculate when both columns exist and sales total is non-zero.

7. PERCENTAGE CONTRIBUTION
Example:
"What percentage of total sales comes from each category?"
Use:
total = df["Sales"].sum()
result = (df.groupby("Category")["Sales"].sum() / total * 100).sort_values(ascending=False)
Only calculate when the requested grouping and metric columns exist.

8. AVERAGE PER ORDER
If OrderID exists and the question asks for average sales per order:
result = df["Sales"].sum() / df["OrderID"].nunique()
Do not divide by row count when a unique order identifier is available.

9. GROUP TOTAL
Example:
"What is the total sales for each category?"
result = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)

10. GROUP AVERAGE
Example:
"What is the average profit for each category?"
result = df.groupby("Category")["Profit"].mean().sort_values(ascending=False)

11. HIGHEST / LOWEST GROUP
Example:
"Which state has the highest sales?"
result = df.groupby("State")["Sales"].sum().sort_values(ascending=False).head(1)

12. TOP N / BOTTOM N
Automatically detect the requested N and calculate the requested ranking.

13. COUNT / UNIQUE COUNT
Use len(df) for total records and nunique() when the question asks
for unique orders, customers, products, etc., provided the relevant
column exists.

14. CORRELATION
Example:
"What is the correlation between sales and profit?"
result = df["Sales"].corr(df["Profit"])
Only when both numerical columns exist.

15. PERCENTAGE CHANGE / GROWTH
Use:
((new_value - old_value) / old_value) * 100
only when the old and new periods can be reliably identified.

16. REVENUE
Use an actual Revenue column only if it exists.
NEVER assume Sales = Revenue.
If Revenue does not exist and cannot be reliably derived from explicit
dataset fields, return CANNOT_DETERMINE.

17. OTHER DERIVED METRICS
For a clearly defined mathematical metric, derive the formula from
the user's question only when all required fields exist and the
meaning is unambiguous.

------------------------------------------------------------
MATHEMATICAL SAFETY RULES
------------------------------------------------------------

- Never divide by zero.
- Never use a column that does not exist.
- Never invent Revenue, Cost, Tax, Salary, Customer Satisfaction,
  Market Share, Growth, or another unavailable metric.
- Never assume Sales = Revenue.
- Never assume Profit = Revenue - Cost unless the required fields
  actually exist.
- For date/growth questions, verify usable date/year information.
- For ambiguous formulas, refuse instead of guessing.
- If required information is missing, return CANNOT_DETERMINE with
  a clear reason.

------------------------------------------------------------
CODE RULES
------------------------------------------------------------

If answerable:

Return executable Python code.

The code MUST:

- use df or tables
- calculate the requested result
- store the final result in a variable named result
- not import anything
- not read files
- not write files
- not access the operating system
- not call APIs
- not use network access
- not use eval or exec
- not use open()
- not contain markdown
- not contain explanations

Example:

result = df["Sales"].sum()

Another example:

result = (
    df.groupby("State")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

------------------------------------------------------------
RESPONSE FORMAT
------------------------------------------------------------

If the question is answerable, return EXACTLY:

STATUS: ANSWERABLE
CODE:
<python code>

If the question cannot be answered reliably, return EXACTLY:

STATUS: CANNOT_DETERMINE
REASON:
<short explanation>

------------------------------------------------------------
USER QUESTION
------------------------------------------------------------

{question}

------------------------------------------------------------
DATASET INFORMATION
------------------------------------------------------------

{data_context}
"""

    return prompt


# =========================================================
# EXTRACT AI RESPONSE
# =========================================================

def extract_ai_response(text):

    text = text.strip()

    upper_text = text.upper()

    # -----------------------------------------------------
    # REFUSAL
    # -----------------------------------------------------

    if "STATUS: CANNOT_DETERMINE" in upper_text:

        reason = ""

        if "REASON:" in text:

            reason = text.split(
                "REASON:",
                1
            )[1].strip()

        if not reason:

            reason = (
                "The question cannot be answered reliably "
                "from the available dataset."
            )

        return {
            "status": "CANNOT_DETERMINE",
            "reason": reason
        }

    # -----------------------------------------------------
    # ANSWERABLE
    # -----------------------------------------------------

    if "STATUS: ANSWERABLE" not in upper_text:

        return {
            "status": "INVALID_AI_RESPONSE",
            "reason": "Gemini returned an unexpected response format."
        }

    if "CODE:" not in text:

        return {
            "status": "INVALID_AI_RESPONSE",
            "reason": "Gemini did not provide executable code."
        }

    code = text.split(
        "CODE:",
        1
    )[1].strip()

    # Remove markdown fences if Gemini accidentally adds them
    code = code.replace("```python", "")
    code = code.replace("```", "")
    code = code.strip()

    return {
        "status": "ANSWERABLE",
        "code": code
    }


# =========================================================
# CODE SAFETY CHECK
# =========================================================

def validate_generated_code(code):

    dangerous_names = {
        "eval",
        "exec",
        "open",
        "compile",
        "__import__",
        "input",
        "globals",
        "locals"
    }

    dangerous_modules = {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "requests",
        "pathlib"
    }

    dangerous_attributes = {
        "to_csv",
        "to_excel",
        "to_pickle",
        "to_sql",
        "read_csv",
        "read_excel",
        "read_pickle",
        "read_json",
        "read_sql",
        "read_html"
    }

    try:

        tree = ast.parse(code)

    except SyntaxError as error:

        return False, f"Generated code has a syntax error: {error}"

    for node in ast.walk(tree):

        # Imports are not allowed
        if isinstance(
            node,
            (ast.Import, ast.ImportFrom)
        ):

            return False, "Generated code contains an import."

        # Dangerous function calls
        if isinstance(node, ast.Call):

            if isinstance(node.func, ast.Name):

                if node.func.id in dangerous_names:

                    return False, (
                        f"Dangerous function '{node.func.id}' "
                        "was detected."
                    )

            if isinstance(node.func, ast.Attribute):

                if node.func.attr in dangerous_attributes:

                    return False, (
                        f"File/network operation "
                        f"'{node.func.attr}' is not allowed."
                    )

        # Dangerous module access
        if isinstance(node, ast.Attribute):

            if isinstance(node.value, ast.Name):

                if node.value.id in dangerous_modules:

                    return False, (
                        f"Access to '{node.value.id}' is not allowed."
                    )

        # Private attributes
        if isinstance(node, ast.Name):

            if node.id.startswith("__"):

                return False, (
                    "Private Python objects are not allowed."
                )

        if isinstance(node, ast.Attribute):

            if node.attr.startswith("__"):

                return False, (
                    "Private Python attributes are not allowed."
                )

    return True, ""


# =========================================================
# RESULT SERIALIZATION
# =========================================================

def normalize_result(value):

    if isinstance(value, pd.DataFrame):

        value = value.where(
            pd.notna(value),
            None
        )

        return value.to_dict(
            orient="records"
        )

    if isinstance(value, pd.Series):

        return {
            str(key): normalize_result(item)
            for key, item in value.items()
        }

    if isinstance(value, dict):

        return {
            str(key): normalize_result(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):

        return [
            normalize_result(item)
            for item in value
        ]

    if isinstance(value, np.generic):

        return value.item()

    if value is pd.NA:

        return None

    if isinstance(value, float):

        if math.isnan(value):

            return None

        if math.isinf(value):

            return str(value)

    return value


# =========================================================
# EXECUTE GENERATED CODE
# =========================================================

def execute_generated_code(code):

    valid, error = validate_generated_code(code)

    if not valid:

        return {
            "success": False,
            "reason": error
        }

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_path = Path(temp_dir)

        # -------------------------------------------------
        # Save all tables as temporary pickle files
        # -------------------------------------------------

        table_paths = {}

        for table_name, df in current_tables.items():

            safe_name = (
                table_name
                .replace("/", "_")
                .replace("\\", "_")
                .replace(" ", "_")
            )

            file_path = (
                temp_path /
                f"{safe_name}.pkl"
            )

            df.to_pickle(file_path)

            table_paths[table_name] = file_path

        # -------------------------------------------------
        # Build table loading code
        # -------------------------------------------------

        table_loading = """
tables = {}
"""

        for table_name, file_path in table_paths.items():

            table_loading += (
                f"tables[{table_name!r}] = "
                f"pd.read_pickle({str(file_path)!r})\n"
            )

        first_table = list(table_paths.keys())[0]

        table_loading += (
            f"df = tables[{first_table!r}]\n"
        )

        # -------------------------------------------------
        # Result serializer inside subprocess
        # -------------------------------------------------

        wrapper_code = f"""
import json
import math
import pandas as pd
import numpy as np

{table_loading}

# =====================================================
# AI GENERATED ANALYSIS
# =====================================================

{code}

# =====================================================
# CHECK RESULT
# =====================================================

if "result" not in locals():
    raise RuntimeError(
        "AI generated code did not create a variable named 'result'."
    )


def clean(value):

    if isinstance(value, pd.DataFrame):

        value = value.where(
            pd.notna(value),
            None
        )

        return value.to_dict(
            orient="records"
        )

    if isinstance(value, pd.Series):

        return {{
            str(key): clean(item)
            for key, item in value.items()
        }}

    if isinstance(value, dict):

        return {{
            str(key): clean(item)
            for key, item in value.items()
        }}

    if isinstance(value, (list, tuple)):

        return [
            clean(item)
            for item in value
        ]

    if isinstance(value, np.generic):

        return value.item()

    if value is pd.NA:

        return None

    if isinstance(value, float):

        if math.isnan(value):

            return None

        if math.isinf(value):

            return str(value)

    return value


final_result = clean(result)

print(
    "__TRUSTWORTHY_RESULT__"
    + json.dumps(
        final_result,
        default=str
    )
)
"""

        script_path = (
            temp_path /
            "analysis_script.py"
        )

        script_path.write_text(
            wrapper_code,
            encoding="utf-8"
        )

        try:

            process = subprocess.run(
                [
                    sys.executable,
                    str(script_path)
                ],
                capture_output=True,
                text=True,
                timeout=20
            )

        except subprocess.TimeoutExpired:

            return {
                "success": False,
                "reason": (
                    "The generated analysis exceeded "
                    "the execution time limit."
                )
            }

        if process.returncode != 0:

            return {
                "success": False,
                "reason": process.stderr.strip()
            }

        marker = "__TRUSTWORTHY_RESULT__"

        if marker not in process.stdout:

            return {
                "success": False,
                "reason": (
                    "The generated code executed but "
                    "did not return a valid result."
                )
            }

        result_text = process.stdout.split(
            marker,
            1
        )[1].strip()

        try:

            result = json.loads(result_text)

        except json.JSONDecodeError:

            result = result_text

        return {
            "success": True,
            "result": result
        }


# =========================================================
# DATA QUALITY
# =========================================================

def get_data_quality():

    quality = {}

    for table_name, df in current_tables.items():

        missing = {
            str(column): int(count)
            for column, count in df.isnull().sum().items()
            if count > 0
        }

        quality[table_name] = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "duplicate_rows": int(
                df.duplicated().sum()
            ),
            "missing_values": missing
        }

    return quality


# =========================================================
# TRUTH SCORE
# =========================================================

def calculate_truth_score(
    stable,
    execution_success
):

    if not execution_success:

        return 0

    score = 100

    total_duplicates = 0
    total_missing = 0

    for table_quality in get_data_quality().values():

        total_duplicates += (
            table_quality["duplicate_rows"]
        )

        total_missing += sum(
            table_quality["missing_values"].values()
        )

    if total_duplicates > 0:

        score -= 10

    if total_missing > 0:

        score -= 5

    if not stable:

        score -= 25

    return max(0, score)


# =========================================================
# ASK QUESTION
# =========================================================

@app.post("/ask")
async def ask_question(question: str):

    global current_df

    # =====================================================
    # CHECK DATASET
    # =====================================================

    if current_df is None:

        return {
            "success": False,
            "status": "CANNOT_DETERMINE",
            "question": question,
            "answer": None,
            "reason": "Please upload a dataset first.",
            "proof_code": None
        }

    # =====================================================
    # BUILD AI PROMPT
    # =====================================================

    prompt = build_ai_prompt(question)

    # =====================================================
    # ASK GEMINI
    # =====================================================

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        ai_text = response.text

    except Exception as error:

        return {
            "success": False,
            "status": "AI_ERROR",
            "question": question,
            "answer": None,
            "reason": str(error),
            "hint": (
                "Check GEMINI_API_KEY and make sure "
                f"{MODEL_NAME} is available."
            )
        }

    # =====================================================
    # PARSE AI RESPONSE
    # =====================================================

    parsed = extract_ai_response(ai_text)

    # =====================================================
    # AI REFUSAL
    # =====================================================

    if parsed["status"] == "CANNOT_DETERMINE":

        return {
            "success": True,
            "status": "CANNOT_DETERMINE",
            "question": question,
            "answer": None,
            "reason": parsed["reason"],
            "proof_code": None,
            "data_quality": get_data_quality(),
            "reliability": "NOT_ANSWERABLE"
        }

    # =====================================================
    # INVALID RESPONSE
    # =====================================================

    if parsed["status"] != "ANSWERABLE":

        return {
            "success": False,
            "status": "AI_CODE_ERROR",
            "question": question,
            "answer": None,
            "reason": parsed["reason"]
        }

    generated_code = parsed["code"]

    # =====================================================
    # FIRST EXECUTION
    # =====================================================

    execution_1 = execute_generated_code(
        generated_code
    )

    if not execution_1["success"]:

        return {
            "success": True,
            "status": "CANNOT_DETERMINE",
            "question": question,
            "answer": None,
            "reason": (
                "The AI-generated analysis could not be "
                "safely executed. The system refused to "
                "return an unverified answer.\n\n"
                + execution_1["reason"]
            ),
            "proof_code": generated_code,
            "data_quality": get_data_quality(),
            "reliability": "UNVERIFIED",
            "truth_score": 0
        }

    # =====================================================
    # SECOND EXECUTION
    # =====================================================
    # Re-run the same generated code.
    # This is our stability verification.

    execution_2 = execute_generated_code(
        generated_code
    )

    if not execution_2["success"]:

        return {
            "success": True,
            "status": "CANNOT_DETERMINE",
            "question": question,
            "answer": None,
            "reason": (
                "The analysis produced a result once but "
                "failed during independent verification."
            ),
            "proof_code": generated_code,
            "data_quality": get_data_quality(),
            "reliability": "UNVERIFIED",
            "truth_score": 0
        }

    # =====================================================
    # STABILITY CHECK
    # =====================================================

    result_1 = execution_1["result"]
    result_2 = execution_2["result"]

    stable = (
        json.dumps(
            result_1,
            sort_keys=True,
            default=str
        )
        ==
        json.dumps(
            result_2,
            sort_keys=True,
            default=str
        )
    )

    # =====================================================
    # TRUTH SCORE
    # =====================================================

    truth_score = calculate_truth_score(
        stable=stable,
        execution_success=True
    )

    # =====================================================
    # FINAL RESPONSE
    # =====================================================

    if not stable:

        return {
            "success": True,
            "status": "CANNOT_DETERMINE",
            "question": question,
            "answer": None,
            "reason": (
                "The analysis result was not stable when "
                "the generated proof code was executed twice."
            ),
            "proof_code": generated_code,
            "data_quality": get_data_quality(),
            "reliability": "UNSTABLE",
            "truth_score": truth_score
        }

    return {
        "success": True,
        "status": "ANSWERABLE",
        "question": question,
        "answer": result_1,
        "proof_code": generated_code,
        "data_quality": get_data_quality(),
        "verification": {
            "code_executed": True,
            "executed_twice": True,
            "stable_result": True
        },
        "reliability": "VERIFIED",
        "truth_score": truth_score
    }
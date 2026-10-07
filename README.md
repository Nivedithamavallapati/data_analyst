# data_analyst
# Trustworthy Data Analyst

## Proof-Carrying AI Data Analysis System

An AI-powered data analyst that answers natural-language questions over real-world datasets while providing executable proof, verification, data-quality information, and a reliability assessment.

Unlike a conventional AI system that may directly generate an answer, Trustworthy Data Analyst generates executable Pandas analysis code, validates the generated code, executes it on the uploaded dataset, executes it again to check stability, and returns the answer together with the proof code.

If the required information is missing, the question is ambiguous, or the generated analysis cannot be reliably verified, the system can return:

"CANNOT DETERMINE"

instead of guessing.

---

# 1. Problem Statement

Traditional AI data analysis systems can produce answers that appear correct but may be unreliable because of:

- Missing data
- Duplicate records
- Contradictory information
- Ambiguous questions
- Missing required columns
- Incorrect mathematical assumptions
- Unsupported metrics
- Hallucinated values
- Unverified calculations

The objective of this project is to build a trustworthy AI data analyst that not only answers questions but also provides evidence that the answer can be reproduced and verified.

---

# 2. Our Solution

Trustworthy Data Analyst follows a proof-carrying analysis approach.

The user uploads a dataset and asks a question in natural language.

The system then:

1. Reads the uploaded dataset.
2. Performs data-quality checks.
3. Sends the question and dataset information to Gemini.
4. Gemini understands the question.
5. Gemini automatically generates Pandas analysis code.
6. The backend validates the generated code.
7. The generated code is executed on the actual dataset.
8. The same analysis is executed a second time.
9. The results are compared for stability.
10. The system calculates a reliability/truth score.
11. The final answer and executable proof code are returned.

If the required data is unavailable or the question cannot be answered reliably, the system refuses to guess.

---

# 3. Key Features

## 3.1 Natural Language Data Analysis

Users can ask questions such as:

- What is the total sales?
- What is the average profit?
- Which state has the highest sales?
- What are the top 5 states by profit?
- What is the ratio of profit to sales?
- What is the profit margin?
- What percentage of sales comes from each category?

The system automatically determines the required analysis instead of relying on hard-coded question keywords.

---

## 3.2 Automatic Code Generation

Gemini generates executable Pandas code based on the user's question and the available dataset.

Example:

User question:

"What is the total sales?"

Generated analysis:

```python
result = df["Sales"].sum()

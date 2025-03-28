# prompts.py

# Prompt template for deciding the next action
DECIDE_ACTION_PROMPT_TEMPLATE = (
    "Analyze the user's request and conversation history. Decide next action.\n"
    "History: {history}\n"
    "Request: {input}\n"
    "Available data: {metadata}\n\n"
    "Options:\n"
    "- code_generation: if the request is a clear technical instruction.\n"
    "- chat_response: for general conversation.\n\n"
    "Return a valid JSON with the key 'action'."
)

# Prompt template for generating a chat response
CHAT_RESPONSE_PROMPT_TEMPLATE = (
    "Answer the following in a friendly and concise manner (under 10 words).\n"
    "User: {input}\n"
    "Response:"
)

# Prompt template for code generation. It uses metadata placeholders.
CODE_GENERATION_PROMPT_TEMPLATE = (
    "You are an expert data scientist. Here is metadata for a DataFrame named 'df':\n"
    "Columns: {columns}\n"
    "Data Types: {dtypes}\n"
    "Sample Rows: {sample_rows}\n"
    "Numerical Ranges: {numerical_ranges}\n"
    "Categorical Values: {categorical_values}\n\n"
    "User command:\n"
    "{command}\n\n"
    "Generate a concise Python snippet that accomplishes the user’s request. "
    "You may use any of these libraries: Pandas, NumPy, Matplotlib, Seaborn, Plotly. "
    "Assume 'df' is already defined, so do NOT write code to read CSV or create 'df'.\n\n"
    "Examples:\n"
    "Example 1:\n"
    "```python\nimport seaborn as sns\nsns.countplot(x='some column', data=df)\nplt.show()\n```\n\n"
    "Example 2:\n"
    "```python\ncount_above_70 = df[df['some column'] > 70].shape[0]\ncount_above_70\n```\n\n"
    "Example 3:\n"
    "```python\nimport plotly.express as px\npx.histogram(df, x='some column', nbins=20)\nfig.show()\n```\n\n"
    "Now generate code for the user command: {command}\n"
    "```python"
)

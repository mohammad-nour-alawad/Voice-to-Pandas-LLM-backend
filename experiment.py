import math
import plotly
import requests
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

CSV_DATASET_PATH = "exps/otto_group_product.csv"
COMMANDS_TXT_PATH = "exps/otto_group_product.txt"
API_URL = "http://localhost:6000/converse"

def replace_nan(data):
    if isinstance(data, dict):
        return {k: replace_nan(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_nan(item) for item in data]
    elif isinstance(data, float) and math.isnan(data):
        return None
    else:
        return data
    
def build_metadata(csv_path: str) -> dict:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    metadata = {}
    metadata['columns'] = list(df.columns)
    metadata['dtypes'] = df.dtypes.apply(lambda x: x.name).to_dict()
    metadata['sample_rows'] = df.head(3).to_dict(orient='records')

    numerical_cols = df.select_dtypes(include=[np.number]).columns
    metadata['numerical_ranges'] = {}
    for col in numerical_cols:
        min_value = df[col].min()
        max_value = df[col].max()
        if isinstance(min_value, (np.integer, np.int64, np.int32)):
            min_value = int(min_value)
        elif isinstance(min_value, (np.floating, np.float64, np.float32)):
            min_value = float(min_value)
        if isinstance(max_value, (np.integer, np.int64, np.int32)):
            max_value = int(max_value)
        elif isinstance(max_value, (np.floating, np.float64, np.float32)):
            max_value = float(max_value)
        metadata['numerical_ranges'][col] = {'min': min_value, 'max': max_value}

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    metadata['categorical_values'] = {}
    for col in categorical_cols:
        unique_values = df[col].unique()
        if len(unique_values) <= 20:
            metadata['categorical_values'][col] = unique_values.tolist()
        else:
            metadata['categorical_values'][col] = unique_values[:20].tolist() + ['...']

    return metadata

def evaluate_generated_code(code: str, df: pd.DataFrame) -> int:
    """
    Executes the generated code in a restricted environment.
    Returns 1 if the code runs successfully, 0 if an error occurs.
    """
    # Set up a restricted environment similar to your Django view.
    allowed_locals = {
        'df': df,
        'pd': pd,
        'np': np,
        'plt': plt,
        'sns': sns,
        'px': px,
        'plotly': plotly
    }
    # Disable built-ins to reduce risk.
    exec_globals = {"__builtins__": {}}
    
    # Remove import statements for security
    cleaned_code_lines = []
    for line in code.split("\n"):
        if not line.strip().startswith("import"):
            cleaned_code_lines.append(line)
    cleaned_code = "\n".join(cleaned_code_lines)
    
    # Attempt to determine if the last line is an expression that should be evaluated
    lines = cleaned_code.strip().split('\n')
    last_line = lines[-1].strip() if lines else ''
    def is_expression(s):
        return (
            not s.startswith(('print', 'plt.', 'sns.', 'px.'))
            and '=' not in s
            and not s.endswith(':')
        )
    
    try:
        if len(lines) > 1 and is_expression(last_line):
            code_body = '\n'.join(lines[:-1])
            exec(code_body, exec_globals, allowed_locals)
            # Evaluate the last line as an expression
            _ = eval(last_line, exec_globals, allowed_locals)
        else:
            exec(cleaned_code, exec_globals, allowed_locals)
        return 1  # Code executed successfully.
    except Exception as e:
        print(f"Execution error: {e}")
        return 0  # Execution failed.

def run_experiment():
    total_score = 0
    total_time = 0
    count = 0

    # Read the dataset once and reuse for evaluation.
    df = pd.read_csv(CSV_DATASET_PATH)
    metadata = build_metadata(CSV_DATASET_PATH)
    metadata = replace_nan(metadata)

    with open(COMMANDS_TXT_PATH, "r") as f:
        commands = [line.strip() for line in f if line.strip()]

    print("Starting experiment...\n")
    for command in commands:
        payload = {
            "user_input": command,
            "metadata": metadata,
            "conversation_history": []
        }

        start_time = time.time()
        try:
            response = requests.post(API_URL, json=payload)
        except Exception as e:
            print(f"Error connecting to API for command '{command}': {e}")
            continue
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        count += 1

        if response.status_code == 200:
            data = response.json()
            generated_code = data.get("code", "")
            execution_score = evaluate_generated_code(generated_code, df) if generated_code else 0
            total_score += execution_score

            print(f"Command: {command}")
            print(f"Execution Score: {execution_score} (1=successful, 0=failed)")
            print(f"Response Time: {elapsed_time:.2f} sec")
            print("-" * 40)
        else:
            print(f"Command: {command} -- API Error: {response.status_code}")
            print("-" * 40)

    average_time = total_time / count if count else 0
    print("\nExperiment Summary: ", CSV_DATASET_PATH)
    print("===================")
    print(f"Total Commands Processed: {count}")
    print(f"Total Execution Score: {total_score} out of {count}")
    print(f"Average Response Time: {average_time:.2f} sec")

if __name__ == "__main__":
    run_experiment()

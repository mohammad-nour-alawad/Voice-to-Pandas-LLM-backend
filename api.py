# api.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import io
import matplotlib.pyplot as plt
import base64
from typing import Union
from pydantic import BaseModel
import numpy as np
from matplotlib.axes import Axes
import re


app = FastAPI()

# Enable CORS for all origins (you can restrict this in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to your frontend's URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Whisper model on GPU if available
print("Loading Whisper model...")
whisper_model = whisper.load_model("base").to(device)  # You can choose 'small', 'base', etc.

# Load Language Model (e.g., GPT-Neo 1.3B) on GPU if available
print("Loading Language Model...")
model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
language_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
language_model.eval()  # Set model to evaluation mode

# Store uploaded data and metadata
data_store = {}

# Pydantic models for request bodies
class CommandRequest(BaseModel):
    command: str

class CodeExecutionRequest(BaseModel):
    code: str

@app.post("/transcribe")
async def transcribe_voice(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)
        # Use Whisper to transcribe audio
        result = whisper_model.transcribe("temp_audio.wav")
        text = result["text"]
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_code")
def generate_code(request: CommandRequest):
    try:
        if "data" not in data_store:
            raise HTTPException(status_code=400, detail="No data uploaded.")

        command = request.command

        # Get metadata
        metadata = data_store['metadata']
        columns = metadata['columns']
        dtypes = metadata['dtypes']
        sample_rows = metadata['sample_rows']
        numerical_ranges = metadata['numerical_ranges']
        categorical_values = metadata['categorical_values']

        # Construct metadata string
        metadata_str = f"Columns: {columns}\n"
        metadata_str += f"Data Types: {dtypes}\n"
        metadata_str += f"Sample Rows: {sample_rows}\n"
        metadata_str += f"Numerical Ranges: {numerical_ranges}\n"
        metadata_str += f"Categorical Values: {categorical_values}\n"

        # Build prompt
        prompt = (
            "You are an expert data scientist. Given the following data information: \n"
            f"{metadata_str}\n\n"
            "Based on the command given, generate a Python Pandas code snippet to perform the task. "
            "Assume that the data is already loaded into a DataFrame named 'df'. "
            "The final output must be stored in a variable named 'result'. "
            "Do not include any code to load or create the DataFrame. \n"
            "here are some examples:\n"
            "Example 1:\n"
            "Command: plot the test preparation course\n"
            "Python Code: ```python result = df['test preparation course'].value_counts().plot(kind='bar')```\n\n"
            
            "Example 2:\n"
            "Command: how many students has math score more than 70\n"
            "Python Code: ```python result = df[df['math score'] > 70].shape[0]```\n\n"
            
                       
            "Example 3:\n"
            "Command: plot the math score\n"
            "Python Code: ```python result = df['math score'].value_counts().plot(kind='bar')```\n\n"
            
            "Now, apply on the following command:\n"
            f"Command: {command}\n"
            "Python Code:"
        )

        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = language_model.generate(
                inputs,
                max_new_tokens=200,
                num_return_sequences=1,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = generated_text[len(prompt):]
        # Try to extract code block between triple backticks
        start = generated_text.find("```python")
        if start == -1:  # If no "```python" delimiter, try just "```"
            start = generated_text.find("```")
        end = generated_text.find("```", start + len("```"))

        if start != -1 and end != -1:
            # Extract code between the delimiters
            code = generated_text[start + len("```python"):end].strip()
        else:
            # Fallback: Use the entire text if no delimiters are found
            code = generated_text.strip()

        code = re.sub(r'^\s*import .*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^\s*from .* import .*$', '', code, flags=re.MULTILINE)
        
        # Remove comments
        code = re.sub(r'#.*', '', code)
        
        # Remove empty lines
        code = "\n".join([line for line in code.splitlines() if line.strip()])
        return {"code": code}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_data")
async def upload_data(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        filename = file.filename
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(contents))
        elif filename.endswith(".txt"):
            df = pd.read_csv(io.BytesIO(contents), delimiter="\t")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")
        data_store["data"] = df

        # Extract metadata
        metadata = {}
        metadata['columns'] = list(df.columns)
        metadata['dtypes'] = df.dtypes.apply(lambda x: x.name).to_dict()
        metadata['sample_rows'] = df.head(3).to_dict(orient='records')

        # For numerical columns, get min/max
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        metadata['numerical_ranges'] = {}
        for col in numerical_cols:
            min_value = df[col].min()
            max_value = df[col].max()
            # Convert NumPy types to native Python types
            if isinstance(min_value, (np.integer, np.int64, np.int32)):
                min_value = int(min_value)
            elif isinstance(min_value, (np.floating, np.float64, np.float32)):
                min_value = float(min_value)
            if isinstance(max_value, (np.integer, np.int64, np.int32)):
                max_value = int(max_value)
            elif isinstance(max_value, (np.floating, np.float64, np.float32)):
                max_value = float(max_value)
            metadata['numerical_ranges'][col] = {'min': min_value, 'max': max_value}

        # For categorical columns, get unique values (up to a limit)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        metadata['categorical_values'] = {}
        for col in categorical_cols:
            unique_values = df[col].unique()
            if len(unique_values) <= 20:
                metadata['categorical_values'][col] = unique_values.tolist()
            else:
                metadata['categorical_values'][col] = unique_values[:20].tolist() + ['...']

        data_store['metadata'] = metadata

        return {"message": "File uploaded successfully.", "metadata": metadata}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute_code")
def execute_code(request: CodeExecutionRequest):
    try:
        # Check if data is uploaded
        if "data" not in data_store or data_store["data"] is None:
            raise HTTPException(status_code=400, detail="No data uploaded.")
        
        # Retrieve the DataFrame
        df = data_store["data"]

        # Define a restricted namespace
        allowed_names = {"df": df, "pd": pd, "plt": plt, "np": np}
        exec_globals = {"__builtins__": None}  # Restrict built-in functions for safety

        # Clean the provided code (strip unnecessary imports)
        code = request.code
        cleaned_code_lines = [
            line for line in code.split("\n") if not line.strip().startswith("import")
        ]
        cleaned_code = "\n".join(cleaned_code_lines)

        # Execute the code in a restricted namespace
        exec(cleaned_code, exec_globals, allowed_names)

        # Handle the result
        if "result" in allowed_names:
            result = allowed_names["result"]

            # If the result is a DataFrame
            if isinstance(result, pd.DataFrame):
                return {"type": "table", "data": result.to_json(orient="split")}

            # If the result is a Matplotlib Figure
            elif isinstance(result, plt.Figure):
                buf = io.BytesIO()
                result.savefig(buf, format="png")
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode("utf-8")
                return {"type": "plot", "data": img_base64}

            # If the result is a Matplotlib Axes
            elif isinstance(result, Axes):
                fig = result.get_figure()
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode("utf-8")
                return {"type": "plot", "data": img_base64}

            # If the result is any other type
            else:
                return {"type": "text", "data": str(result)}

        # If no 'result' variable is found
        else:
            return {"type": "text", "data": "Code executed successfully but no 'result' variable was found."}

    except KeyError as e:
        # Handle KeyError if 'result' or any variable is not found
        return {"type": "error", "data": f"KeyError: {str(e)}. Check your code logic and variable usage."}

    except TypeError as e:
        # Handle TypeError, likely due to NoneType
        return {"type": "error", "data": f"TypeError: {str(e)}. Ensure your code initializes required variables."}

    except Exception as e:
        # Catch all other errors
        return {"type": "error", "data": f"Error: {str(e)}. Check your code and try again."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6000)

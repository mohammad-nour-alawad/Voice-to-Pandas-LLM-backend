from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import whisper
import torch
from typing import Dict, Any
from pydantic import BaseModel
import re
import time

from vllm import LLM, SamplingParams

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading Whisper model...")
whisper_model = whisper.load_model("turbo", device=device)

print("Loading Language Model with vLLM...")
model_name = "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"

llm = LLM(model_name)

class CodeGenRequest(BaseModel):
    command: str
    metadata: Dict[str, Any]

@app.post("/transcribe")
async def transcribe_voice(file: UploadFile = File(...)):
    """
    Endpoint to transcribe audio using Whisper.
    """
    try:
        audio_bytes = await file.read()
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)
        result = whisper_model.transcribe("temp_audio.wav")
        text = result["text"]
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_code")
def generate_code(request: CodeGenRequest):
    """
    Endpoint to generate Python code from user command and metadata.
    We'll use vLLM for language generation.
    We'll also measure and return the latency of the generation step.
    """
    try:
        start_time = time.perf_counter()

        command = request.command
        metadata = request.metadata

        columns = metadata.get('columns', [])
        dtypes = metadata.get('dtypes', {})
        sample_rows = metadata.get('sample_rows', [])
        numerical_ranges = metadata.get('numerical_ranges', {})
        categorical_values = metadata.get('categorical_values', {})

        # Summarize the metadata
        metadata_str = (
            f"Columns: {columns}\n"
            f"Data Types: {dtypes}\n"
            f"Sample Rows: {sample_rows}\n"
            f"Numerical Ranges: {numerical_ranges}\n"
            f"Categorical Values: {categorical_values}\n"
        )

        prompt = (
            "You are an expert data scientist. Here is metadata for a DataFrame named 'df':\n"
            f"{metadata_str}\n\n"
            "User command:\n"
            f"{command}\n\n"
            "Generate a concise Python snippet that accomplishes the user’s request. "
            "You may use any of these libraries: Pandas, NumPy, Matplotlib, Seaborn, Plotly. "
            "Assume 'df' is already defined, so do NOT write code to read CSV or create 'df'. "
            "You can do table manipulations or create a plot.\n\n"
            "Here are some short examples:\n\n"
            "Example 1:\n"
            "```python\n"
            "import seaborn as sns\n"
            "sns.countplot(x='some column', data=df)\n"
            "plt.show()\n"
            "```\n\n"
            "Example 2:\n"
            "```python\n"
            "count_above_70 = df[df['some column'] > 70].shape[0]\n"
            "count_above_70\n"
            "```\n\n"
            "Example 3:\n"
            "```python\n"
            "import plotly.express as px\n"
            "px.histogram(df, x='some column', nbins=20)\n"
            "fig.show()\n"
            "```\n\n"
            "Now generate code for the user command: "
            f"{command}\n"
            "```python"
        )
        
        
        

        sampling_params = SamplingParams(
            max_tokens=300,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            stop='```'
        )

        outputs = llm.generate([prompt], sampling_params=sampling_params)
        
        generated_text = outputs[0].outputs[0].text

        # Remove the prompt portion from the front if needed.
        # E.g., you might do something like:
        # generated_text = generated_text.replace(prompt, "")
        # or a more robust approach if the model repeats the prompt.

        # Attempt to extract the code block between triple backticks
        start_idx = generated_text.find("```")
        end_idx = generated_text.find("```", start_idx + 3) if start_idx != -1 else -1

        if start_idx != -1 and end_idx != -1:
            code = generated_text[start_idx+3:end_idx].strip()
        else:
            code = generated_text.strip()

        # Clean up multiline imports or anything else if desired
        # code = re.sub(r'^\s*import .*$', '', code, flags=re.MULTILINE)
        # code = re.sub(r'^\s*from .* import .*$', '', code, flags=re.MULTILINE)

        # Remove leftover backticks
        code = code.replace("```", "")

        end_time = time.perf_counter()
        latency = end_time - start_time

        return {
            "code": code,
            "latency_seconds": latency
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6000)

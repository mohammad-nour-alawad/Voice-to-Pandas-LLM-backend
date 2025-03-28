# AI Voice Assistant for Data Visualization & Manipulation

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.0.5+-orange.svg)
![vLLM](https://img.shields.io/badge/vLLM-0.2.5+-yellow.svg)

Voice-enabled AI assistant that helps with data visualization and manipulation through natural language commands, powered by state-of-the-art language models and **LangGraph** workflows.

## 🛠️ Technical Stack

### Core Components
- **Backend**: `FastAPI` (Python)
- **Workflow Engine**: `LangGraph`
- **LLM Serving**: `vLLM`
- **Speech-to-Text**: `Whisper` (medium.en)
- **Text-to-Speech**: `VITS` (VCTK voices)

### Models
| Component | Model | Specification |
|-----------|-------|---------------|
| STT | Whisper | `medium.en` |
| LLM | Qwen2.5-Coder | `32B-Instruct-AWQ` |
| TTS | VITS | `tts_models/en/vctk/vits` (speaker p225) |


## ✨ Key Features

- **Voice Interface**: Speech-to-text and text-to-speech capabilities
- **Intelligent Workflows**: LangGraph-powered decision making
- **Code Generation**: Automatic Python code generation for data tasks
- **Conversational AI**: Context-aware chat responses
- **High Performance**: Optimized inference with vLLM


## 🗂 Project Structure
```bash
ai-voice-assistant/
├── api.py # FastAPI endpoints
├── models.py # Model loading and inference
├── prompts.py # Prompt templates
├── schemas.py # Type definitions and Pydantic models
├── workflow.py # LangGraph workflow definition
└── README.md
```

## 🛠️ Workflow Graph:

```mermaid
graph TD
    A[Client] --> B{{/converse}}
    A --> C{{/transcribe}}
    
    B --> D[Create Workflow]
    D --> E[decide_action]
    E --> F{Decision}
    
    F -->|code_generation| G[generate_code_node]
    F -->|chat_response| H[generate_chat_response_node]
    
    G --> I[[LLM Model]]
    H --> J[[TTS Model]]
    C --> K[[Whisper Model]]
    
    I --> L[Generated Code]
    J --> M[Audio Response]
    K --> N[Transcribed Text]
    
    L --> O{API Response}
    M --> O
    N --> P{{Transcription Result}}
    
    O --> A
    P --> A
    
    style A fill:#4A90E2,color:white
    style B fill:#50E3C2,color:#333
    style C fill:#50E3C2,color:#333
    style E fill:#F5A623,color:white
    style F fill:#FF6B6B,color:white
```


## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- NVIDIA GPU with CUDA support at least 30 GB (I used `NVIDIA RTX 6000 Ada Generation 48GB VRAM`) 
- Docker (recommended)

### Installation
```bash
git clone https://github.com/your-repo/Voice-to-Pandas-LLM-backend.git
cd Voice-to-Pandas-LLM-backend
pip install -r requirements.txt
```

then run the API using:
```bash
CUDA_VISIBLE_DEVICES=0 uvicorn api:app --host 0.0.0.0 --port 6000
```

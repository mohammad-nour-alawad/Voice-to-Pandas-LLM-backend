# models.py

import torch
import whisper
from vllm import LLM
from TTS.api import TTS
from dotenv import load_dotenv
import os
import io

load_dotenv()

STT_MODEL = os.getenv("STT_MODEL")
LLM_MODEL_NAME = os.getenv("LLM_MODEL")
TTS_MODEL_NAME = os.getenv("TTS_MODEL")
TTS_SPEAKER = os.getenv("TTS_SPEAKER")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading Whisper model...")
whisper_model = whisper.load_model(STT_MODEL, device=device)


# models.py

from transformers import AutoTokenizer, AutoConfig

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL_NAME,
    use_fast=True,
    padding_side="left"
)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

llm_config = AutoConfig.from_pretrained(LLM_MODEL_NAME)

print("Loading Language Model with vLLM...")
llm = LLM(
    model=LLM_MODEL_NAME,
    tokenizer=LLM_MODEL_NAME,
    enforce_eager=True,
    max_model_len=llm_config.max_position_embeddings
)

print("Initializing TTS...")
tts = TTS(model_name=TTS_MODEL_NAME, progress_bar=True)

def text_to_speech(text: str) -> bytes:
    audio_buffer = io.BytesIO()
    tts.tts_to_file(text=text, file_path=audio_buffer, speaker=TTS_SPEAKER, speed=1.9)
    audio_buffer.seek(0)
    return audio_buffer.read()

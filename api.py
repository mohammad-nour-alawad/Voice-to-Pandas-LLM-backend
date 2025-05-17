# api.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time

# Import workflow and schemas
from workflow import create_workflow
from models import whisper_model
from schemas import ConversationRequest

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/converse")
async def converse(request: ConversationRequest):
    overall_start = time.perf_counter()
    try:
        workflow = create_workflow()
        initial_state = {
            "user_input": request.user_input,
            "metadata": request.metadata,
            "conversation_history": request.conversation_history,
            "generated_code": None,
            "response_message": None,
            "response_audio": None,
            "decision": None,
            "timing_info": {}
        }
        result = workflow.invoke(initial_state)

        overall_end = time.perf_counter()
        total_time_sec = round(overall_end - overall_start, 4)

        # Gather node timings from "timing_info"
        node_times = result.get("timing_info", {})
        node_times["overall_request_sec"] = total_time_sec

        response_data = {
            "code": result.get("generated_code"),
            "message": result.get("response_message"),
            "audio": result.get("response_audio"),
            "updated_history": result["conversation_history"] + [{
                "user": request.user_input,
                "system": result.get("generated_code") or result.get("response_message")
            }],
            "timing": node_times
        }
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe")
async def transcribe_voice(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)
        result = whisper_model.transcribe("temp_audio.wav")
        return {"text": result["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6000)

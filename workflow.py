# workflow.py

import time
import base64
from typing import Any, Dict
from vllm import SamplingParams
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Import prompt templates and schemas
from prompts import DECIDE_ACTION_PROMPT_TEMPLATE, CHAT_RESPONSE_PROMPT_TEMPLATE, CODE_GENERATION_PROMPT_TEMPLATE
from schemas import AgentState, Decision
from models import llm, text_to_speech

def decide_action(state: AgentState) -> AgentState:
    parser = JsonOutputParser(pydantic_object=Decision)
    prompt = ChatPromptTemplate.from_template(DECIDE_ACTION_PROMPT_TEMPLATE)
    try:
        formatted_prompt = prompt.format(
            input=state["user_input"],
            history=state["conversation_history"],
            metadata=state["metadata"]
        )
    except Exception as e:
        formatted_prompt = f"User request: {state['user_input']}"

    def llm_run(text: str) -> str:
        sampling_params = SamplingParams(max_tokens=150, temperature=0.5)
        outputs = llm.generate([text], sampling_params)
        return outputs[0].outputs[0].text.strip()

    try:
        llm_response = llm_run(formatted_prompt)
        decision = parser.parse(llm_response)
    except Exception as e:
        decision = {"action": "chat_response"}

    state["decision"] = decision
    return state

def route_action(state: AgentState) -> str:
    try:
        return state["decision"]["action"]
    except Exception:
        return "chat_response"

def generate_code_internal(command: str, metadata: Dict[str, Any]):
    start_time = time.perf_counter()
    
    columns = metadata.get('columns', [])
    dtypes = metadata.get('dtypes', {})
    sample_rows = metadata.get('sample_rows', [])
    numerical_ranges = metadata.get('numerical_ranges', {})
    categorical_values = metadata.get('categorical_values', {})

    formatted_prompt = CODE_GENERATION_PROMPT_TEMPLATE.format(
        columns=columns,
        dtypes=dtypes,
        sample_rows=sample_rows,
        numerical_ranges=numerical_ranges,
        categorical_values=categorical_values,
        command=command
    )
    
    sampling_params = SamplingParams(
        max_tokens=300,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        stop='```'
    )

    outputs = llm.generate([formatted_prompt], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text

    start_idx = generated_text.find("```")
    end_idx = generated_text.find("```", start_idx + 3) if start_idx != -1 else -1

    if start_idx != -1 and end_idx != -1:
        code = generated_text[start_idx+3:end_idx].strip()
    else:
        code = generated_text.strip()

    code = code.replace("```", "")
    end_time = time.perf_counter()
    latency = end_time - start_time

    return {
        "code": code,
        "latency_seconds": latency
    }

def generate_code_node(state: AgentState) -> AgentState:
    code_info = generate_code_internal(state["user_input"], state["metadata"])
    state["generated_code"] = code_info.get("code")
    state["response_message"] = "Code generated, please execute!"
    return state

def generate_chat_response_node(state: AgentState) -> AgentState:
    prompt = CHAT_RESPONSE_PROMPT_TEMPLATE.format(input=state["user_input"])
    sampling_params = SamplingParams(max_tokens=20, temperature=0.3, top_p=0.9, top_k=50)
    outputs = llm.generate([prompt], sampling_params)
    response = outputs[0].outputs[0].text.strip()
    state["response_message"] = response

    try:
        audio_bytes = text_to_speech(response)
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        state["response_audio"] = audio_base64
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        state["response_audio"] = None

    return state

def create_workflow():
    builder = StateGraph(AgentState)
    builder.add_node("decide_action", decide_action)
    builder.add_node("generate_code", generate_code_node)
    builder.add_node("generate_chat_response", generate_chat_response_node)
    builder.add_conditional_edges(
        "decide_action",
        route_action,
        {
            "code_generation": "generate_code",
            "chat_response": "generate_chat_response"
        }
    )
    builder.add_edge("generate_code", END)
    builder.add_edge("generate_chat_response", END)
    builder.set_entry_point("decide_action")
    return builder.compile()

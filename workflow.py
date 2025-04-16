# workflow.py

import time
import base64
from typing import Any, Dict
from vllm import SamplingParams
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Import prompt templates and schemas
from prompts import DECIDE_ACTION_PROMPT, CHAT_RESPONSE_PROMPT, CODE_GENERATION_PROMPT
from schemas import AgentState, Decision
from models import llm, tokenizer, text_to_speech


def format_prompt(messages_template: list, state: AgentState, metadata_fields: dict = None) -> str:
    """Format chat template with current state and metadata"""
    formatted_messages = []
    
    for msg in messages_template:
        content = msg["content"].format(
            input=state["user_input"],
            history=state["conversation_history"],
            metadata=state["metadata"],
            **(metadata_fields or {})
        )
        formatted_messages.append({"role": msg["role"], "content": content})
    
    return tokenizer.apply_chat_template(
        formatted_messages,
        tokenize=False,
        add_generation_prompt=True
    )


def decide_action(state: AgentState) -> AgentState:
    """Decision node with enhanced error handling"""
    parser = JsonOutputParser(pydantic_object=Decision)
    
    try:
        prompt = format_prompt(DECIDE_ACTION_PROMPT, state)
        sampling_params = SamplingParams(
            max_tokens=200,
            temperature=0.3,
            stop=["</s>", "\n\n"]
        )
        
        #print("decide_prompt:\n", prompt)
        outputs = llm.generate([prompt], sampling_params)
        raw_response = outputs[0].outputs[0].text.strip()
        decision = parser.parse(raw_response)
    except Exception as e:
        print(f"Decision error: {str(e)}")
        decision = {"action": "chat_response"}
    
    state["decision"] = decision
    return state


def route_action(state: AgentState) -> str:
    try:
        return state["decision"]["action"]
    except Exception:
        return "chat_response"


def generate_code_node(state: AgentState) -> AgentState:
    """Code generation with structured metadata handling"""
    metadata = state["metadata"]
    code_prompt = format_prompt(
        CODE_GENERATION_PROMPT,
        state,
    )
    #print("code_prompt:\n", code_prompt)
    
    sampling_params = SamplingParams(
        max_tokens=400,
        temperature=0.2,
        top_p=0.95,
        stop=["<|", "</s>"]
    )
    
    start_time = time.perf_counter()
    outputs = llm.generate([code_prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text
    latency = time.perf_counter() - start_time
    
    # Extract code block
    code_block = generated_text.split("```python")[-1].split("```")[0].strip()
    
    state.update({
        "generated_code": code_block,
        "response_message": "Here's the generated code:",
        "latency": f"{latency:.2f}s"
    })
    return state

def generate_chat_response_node(state: AgentState) -> AgentState:
    """Chat response generation with TTS integration"""
    chat_prompt = format_prompt(CHAT_RESPONSE_PROMPT, state)
    
    sampling_params = SamplingParams(
        max_tokens=200,
        temperature=0.7,
        top_p=0.9,
        stop=["</s>"]
    )
    
    
    #print("chat_prompt:\n", chat_prompt)
    outputs = llm.generate([chat_prompt], sampling_params)
    response = outputs[0].outputs[0].text.strip()
    
    # Generate audio response
    try:
        audio_bytes = text_to_speech(response)
        state["response_audio"] = base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        state["response_audio"] = None
    
    state["response_message"] = response
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

import requests
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Digital Fridge LLM Gateway")

SYSTEM_PROMPT = """You are a strict data formatting machine for a smart fridge simulator.
Your ONLY job is to convert incoming user actions, item lists, and context parameters into a clean string vector line packed inside the "vector" key of a JSON object.

STRICT ACTION PATTERNS:
1. If Action is "shopping":
   Output exactly X items using format -> SHOPPING,CATEGORY:NAME:PRICE:PORTIONS:WEIGHT_GRAMS,...
   Example: {"vector": "SHOPPING,1:CHICKEN FILLET:3100:2:2434.5,2:TOMATOES:924:1:415.2"}

2. If Action is "cooking":
   Output format -> COOKING,DISH_NAME:YIELD_PORTIONS:YIELD_WEIGHT,USED_ID:USED_PORTIONS:USED_WEIGHT,...
   Example: {"vector": "COOKING,CHICKEN SOUP:4:2000.0,1:1.0:1200.0,2:2.0:412.0"}

3. If Action is "consumption":
   Output format -> CONSUMPTION,LOCAL_ID:PORTIONS:WEIGHT_GRAMS,...
   Example: {"vector": "CONSUMPTION,1:1.0:450.5,2:0.5:250.0"}

4. If Action is "waste":
   Output format -> WASTE,LOCAL_ID:PORTIONS:WEIGHT:REASON_CODE,... (Reason codes: 1=Mold, 2=Stale, 3=Expired, 4=Smell)
   Example: {"vector": "WASTE,1:1.0:412.0:1,2:0.5:250.0:3"}"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "vector": {
            "type": "string",
            "description": "The exact flat data string vector following the blueprint pattern."
        }
    },
    "required": ["vector"]
}

# We structure the payload properties explicitly here
class InferenceRequest(BaseModel):
    action: str      # e.g., "shopping", "cooking", "consumption", "waste"
    items: str       # e.g., "3" or "1:EGG(67),2:MILK(289)" or empty
    context: str     # e.g., "1:2:3:0:1:4:2:1:0:0:5:2:1:9:0:1:1:3:2:4"

@app.post("/generate")
def generate_behavior(request: InferenceRequest):
    try:
        # Build an easy-to-read, clean task prompt for Ollama
        user_instruction = (
            f"ACTION TYPE: {request.action}\n"
            f"INPUT ITEMS DATA: {request.items}\n"
            f"RESIDENT CONTEXT VECTOR: {request.context}\n\n"
            f"Convert this data into the strict '{request.action}' flat vector line inside the 'vector' key now."
        )

        response = requests.post('http://ollama:11434/api/chat', json={
            "model": "qwen2.5:3b", 
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_instruction}
            ],
            "format": RESPONSE_SCHEMA,
            "options": {
                "temperature": 0.0,     # Enforces zero formatting drift
                "num_thread": 4,        # Locks to your 4 fast Intel P-cores
                "num_ctx": 2048,
                "num_predict": 120,
                "stop": ["\n", "User:", "Assistant:"]
            },
            "stream": False
        })
        
        ollama_json = response.json()['message']['content']
        data = json.loads(ollama_json)
        return {"vector_output": data.get("vector", "").strip()}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference engine failure: {str(e)}")
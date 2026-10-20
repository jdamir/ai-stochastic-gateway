import requests
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Digital Fridge LLM Gateway")

# Flat vector layout using ONLY colons and commas across all 4 actions
SYSTEM_PROMPT = """You are a strict data formatting machine for a smart fridge simulator.
Convert the input query into a single raw string line and return it inside the "vector" key of a JSON object.

STRICT FORMAT RULES:
- Return ONLY JSON: {"vector": "YOUR_FLAT_STRING_HERE"}
- Write everything in ONE single line.
- Do NOT include spaces after commas. No markdown blockers.

ACTION BLUEPRINTS:
1. For "shopping|X|...":
   Pattern -> SHOPPING,CATEGORY:NAME:PRICE:PORTIONS:WEIGHT,...
   Example: {"vector": "SHOPPING,1:CHICKEN FILLET:3100:2:2434.5,2:TOMATOES:924:1:415.2"}

2. For "cooking|AVAILABLE_ITEMS|...":
   Pattern -> COOKING,DISH_NAME:YIELD_PORTIONS:YIELD_WEIGHT,USED_ID:USED_PORTIONS:USED_WEIGHT,USED_ID:USED_PORTIONS:USED_WEIGHT...
   Example: {"vector": "COOKING,CHICKEN SOUP:4:2000.0,1:1.0:1200.0,2:2.0:412.0"}

3. For "consumption|AVAILABLE_FOOD|...":
   Pattern -> CONSUMPTION,LOCAL_ID:PORTIONS:WEIGHT_GRAMS,LOCAL_ID:PORTIONS:WEIGHT_GRAMS...
   Example: {"vector": "CONSUMPTION,1:1.0:450.5,2:0.5:250.0"}

4. For "waste|SPOILED_LIST":
   Pattern -> WASTE,LOCAL_ID:PORTIONS:WEIGHT:REASON_CODE,LOCAL_ID:PORTIONS:WEIGHT:REASON_CODE...
   Example: {"vector": "WASTE,1:1.0:412.0:1,2:0.5:250.0:3"}"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "vector": {
            "type": "string",
            "pattern": r"^[A-Za-z0-9_\s\.\,\:]+$",  # Completely removed "|" from permitted characters
            "description": "The exact flat data string matching the requested pattern."
        }
    },
    "required": ["vector"]
}

class InferenceRequest(BaseModel):
    query: str

@app.post("/generate")
def generate_behavior(request: InferenceRequest):
    try:
        response = requests.post('http://ollama:11434/api/chat', json={
            "model": "qwen2.5:3b", 
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Convert this query to raw vector data now: {request.query}"}
            ],
            "format": RESPONSE_SCHEMA,
            "options": {
                "temperature": 0.0,      # Strict mathematical extraction
                "num_thread": 4,         # Utilizes your physical Intel P-cores
                "num_ctx": 2048,
                "num_predict": 120,      # Prevents long running CPU loops
                "stop": ["\n", "User:", "Assistant:"]
            },
            "stream": False
        })
        
        ollama_json = response.json()['message']['content']
        data = json.loads(ollama_json)
        return {"vector_output": data.get("vector", "").strip()}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
import requests
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Digital Fridge LLM Gateway")

SYSTEM_PROMPT = """You are a strict data formatting machine for a smart fridge simulator.
Convert the input query into a single raw string line and return it inside the "vector" key of a JSON object.

STRICT FORMAT RULES:
- Return ONLY JSON matching this format: {"vector": "YOUR_FLAT_STRING_HERE"}
- Write everything in ONE single line.
- Do NOT include spaces after commas. No markdown blocks.

ACTION BLUEPRINTS:
1. For "shopping|X|...":
   Pattern -> SHOPPING,CATEGORY:NAME:PRICE:PORTIONS:WEIGHT,...
   Example: {"vector": "SHOPPING,1:CHICKEN FILLET:3100:2:2434.5,2:TOMATOES:924:1:415.2"}

2. For "cooking|AVAILABLE_ITEMS|...":
   Pattern -> COOKING,DISH_NAME:YIELD_PORTIONS:YIELD_WEIGHT,USED_ID:USED_PORTIONS:USED_WEIGHT,...
   Example: {"vector": "COOKING,CHICKEN SOUP:4:2000.0,1:1.0:1200.0,2:2.0:412.0"}

3. For "consumption|AVAILABLE_FOOD|...":
   Pattern -> CONSUMPTION,LOCAL_ID:PORTIONS:WEIGHT_GRAMS,...
   Example: {"vector": "CONSUMPTION,1:1.0:450.5,2:0.5:250.0"}

4. For "waste|SPOILED_LIST":
   Pattern -> WASTE,LOCAL_ID:PORTIONS:WEIGHT:REASON_CODE,...
   Example: {"vector": "WASTE,1:1.0:412.0:1,2:0.5:250.0:3"}"""

# Simplified schema without regex to let the model generate smoothly without syntax rejections
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "vector": {
            "type": "string",
            "description": "The flat data string following the blueprint."
        }
    },
    "required": ["vector"]
}

class InferenceRequest(BaseModel):
    query: str

@app.post("/generate")
def generate_behavior(request: InferenceRequest):
    try:
        raw_response = requests.post('http://ollama:11434/api/chat', json={
            "model": "qwen2.5:3b", 
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Convert this query to raw vector data now: {request.query}"}
            ],
            "format": RESPONSE_SCHEMA,
            "options": {
                "temperature": 0.0,      # Absolute deterministic output
                "num_thread": 4,         # Utilizes your physical Intel P-cores
                "num_ctx": 2048,
                "num_predict": 120,
                "stop": ["\n", "User:", "Assistant:"]
            },
            "stream": False
        })
        
        resp_json = raw_response.json()
        
        # Safe error trapping to catch exactly what Ollama responds with
        if 'message' not in resp_json:
            raise HTTPException(status_code=500, detail=f"Ollama raw error payload: {json.dumps(resp_json)}")
            
        ollama_json = resp_json['message']['content']
        data = json.loads(ollama_json)
        return {"vector_output": data.get("vector", "").strip()}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

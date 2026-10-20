import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Digital Fridge LLM Gateway")

# We simplified and focused the instructions so a 3B model can execute them instantly on CPU
SYSTEM_PROMPT = """You are a strict backend data formatting engine for a smart fridge simulator.
You receive a text query and MUST convert it into a raw string vector.

CRITICAL RULES:
- Output ONLY the raw result. No explanations, no conversation, no markdown wrappers, no backticks.
- Write everything in a SINGLE line.
- Use uppercase for item names.

FORMAT PATTERNS TO FOLLOW EXACTLY:
1. If query starts with "shopping|X|...":
   Output exactly X items using format -> CATEGORY_CODE:ITEM_NAME:PRICE:PORTIONS:WEIGHT_GRAMS,...
   (Category codes: 1=Meat, 2=Vegetables, 3=Dairy, 4=Drinks, 5=Bakery, 6=Sweets)
   Example: 1:CHICKEN FILLET:3100:2:2434.5,2:TOMATOES:924:1:415.2

2. If query starts with "cooking|AVAILABLE_ITEMS|...":
   Output format -> DISH_NAME:YIELD_PORTIONS:YIELD_WEIGHT|LOCAL_ID:USED_PORTIONS:USED_WEIGHT,...
   Example: CHICKEN SOUP WITH TOMATOES:4:2000.0|1:1:1200.0,2:2:412.0

3. If query starts with "consumption|AVAILABLE_FOOD|...":
   Output format -> LOCAL_ID:PORTIONS:WEIGHT_GRAMS,...
   Example: 1:1:450.5,2:0.5:250.0

4. If query starts with "waste|SPOILED_LIST":
   Output format -> LOCAL_ID:PORTIONS:WEIGHT:REASON_CODE,...
   (Reason codes: 1=Mold, 2=Stale, 3=Expired, 4=Smell)
   Example: 1:1.0:412.0:1,2:0.5:250.0:3"""

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
            "options": {
                "temperature": 0.2,     # CRITICAL: Lowered to 0.2 to stop hallucinations and fix structure!
                "num_thread": 4,        # Locks onto your 4 high-performance physical cores
                "num_ctx": 2048,
                "num_predict": 120,     # Prevents long-running loops
                "stop": ["\n", "User:", "Assistant:", "query", "cooking|", "shopping|", "consumption|"] # Hard stops
            },
            "stream": False
        })
        
        result_text = response.json()['message']['content']
        
        # Clean any accidental junk or markdown wrapper attempts
        clean_output = result_text.replace("```", "").replace("text", "").replace("\n", "").strip()
        
        return {"vector_output": clean_output}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama connection error: {str(e)}")
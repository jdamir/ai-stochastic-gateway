import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Digital Fridge LLM Gateway")

SYSTEM_PROMPT = """Role: You are a strict stochastic AI module predicting digital fridge user behavior.
You receive a 20-position context vector (format P1:P2:...:P20). 
P1 (Archetype): 1=Athlete, 2=Student, 3=Family, 4=Workaholic. P2 (Mood): 1=Happy, 2=Tired, 3=Sad. P5 (Income): 1=Low, 2=Medium, 3=High.

STRICT FORMATTING RULES:
- Output ONLY a single line of raw data.
- Do NOT include any introduction, explanations, markdown, or trailing newlines.
- Separate items strictly by commas. No spaces after commas.

ACTION DEFINITIONS:
1. "shopping|X|VECTOR" -> Output exactly X random grocery items matching the profile.
   Category codes: 1=Meat, 2=Vegetables, 3=Dairy, 4=Drinks, 5=Bakery, 6=Sweets.
   Format: CATEGORY_CODE:ITEM_NAME:UNIT_PRICE:PORTIONS:WEIGHT_GRAMS
   Example Response for X=2: 1:CHICKEN FILLET:3100:2:2434.5,2:TOMATOES:924:1:415.2

2. "cooking|AVAILABLE_LIST|VECTOR" -> Cook a single dish from items.
   Format: DISH_NAME:YIELD_PORTIONS:YIELD_WEIGHT|LOCAL_ID:USED_PORTIONS:USED_WEIGHT,...

3. "consumption|AVAILABLE_FOOD|VECTOR" -> Eat existing items by local ID.
   Format: LOCAL_ID:PORTIONS:WEIGHT_GRAMS,...

4. "waste|SPOILED_LIST" -> Throw away items. Reason codes: 1=Mold, 2=Stale, 3=Expired, 4=Smell.
   Format: LOCAL_ID:PORTIONS:WEIGHT:REASON_CODE,..."""

class InferenceRequest(BaseModel):
    query: str

@app.post("/generate")
def generate_behavior(request: InferenceRequest):
    try:
        response = requests.post('http://ollama:11434/api/chat', json={
            "model": "qwen2.5:3b", 
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.query}
            ],
            "options": {
                "temperature": 0.7,     # Slightly lowered for better rule adherence
                "num_thread": 4,        # Uses your physical Intel P-cores
                "num_ctx": 2048,        # Optimized context window
                "num_predict": 150,     # Reduced since vectors are short anyway
                "stop": ["\n", "User:", "Assistant:"] # CRITICAL: Tells the CPU to stop generating immediately
            },
            "stream": False
        })
        
        result_text = response.json()['message']['content']
        # Strip any accidental leading/trailing whitespaces or markdown codeblocks
        clean_output = result_text.replace("```", "").replace("text", "").strip()
        return {"vector_output": clean_output}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Digital Fridge LLM Gateway")

SYSTEM_PROMPT = """You are a strict data formatting robot for a smart fridge simulator.
Your ONLY job is to convert the input query into a compact string vector and return it inside a JSON object with the key "vector".

STRICT OUTPUT FORMAT RULES:
- Return a JSON object: {"vector": "YOUR_COMPACT_STRING_HERE"}
- Do NOT add spaces after commas.
- Do NOT include markdown blocks or conversational text.

PATTERNS TO FOLLOW EXACTLY:
1. For "shopping|X|...":
   Format -> CATEGORY_CODE:ITEM_NAME:PRICE:PORTIONS:WEIGHT_GRAMS,...
   Example: {"vector": "1:CHICKEN FILLET:3100:2:2434.5,2:TOMATOES:924:1:415.2"}

2. For "cooking|AVAILABLE_ITEMS|...":
   Format -> DISH_NAME:YIELD_PORTIONS:YIELD_WEIGHT|LOCAL_ID:USED_PORTIONS:USED_WEIGHT,...
   Example: {"vector": "CHICKEN SOUP WITH TOMATOES:4:2000.0|1:1:1200.0,2:2:412.0"}

3. For "consumption|AVAILABLE_FOOD|...":
   Format -> LOCAL_ID:PORTIONS:WEIGHT_GRAMS,...
   Example: {"vector": "1:1:450.5,2:0.5:250.0"}

4. For "waste|SPOILED_LIST":
   Format -> LOCAL_ID:PORTIONS:WEIGHT:REASON_CODE,... (Reason codes: 1=Mold, 2=Stale, 3=Expired, 4=Smell)
   Example: {"vector": "1:1.0:412.0:1,2:0.5:250.0:3"}"""

# Описываем жесткую схему ответа для Ollama, чтобы она не могла ошибиться в синтаксисе
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "vector": {
            "type": "string",
            "description": "The raw data vector line matching the requested pattern exactly."
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
                {"role": "user", "content": f"Format this query into the 'vector' field now: {request.query}"}
            ],
            "format": RESPONSE_SCHEMA,  # Включаем жесткий JSON-режим Ollama
            "options": {
                "temperature": 0.1,     # Минимальная температура для максимальной точности
                "num_thread": 4,        # Задействуем 4 производительных P-ядра i5
                "num_ctx": 2048,
                "num_predict": 100
            },
            "stream": False
        })
        
        # Получаем JSON от Ollama, парсим его и отдаем симулятору чистую строку
        ollama_json = response.json()['message']['content']
        data = json.loads(ollama_json)
        
        return {"vector_output": data.get("vector", "").strip()}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
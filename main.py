import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Digital Fridge LLM Gateway")

SYSTEM_PROMPT = """Role: You are a stochastic AI module predicting digital fridge user behavior.
Context profile matrix (20 positions, format P1:P2:...:P20):
P1 (Archetype): 1=Athlete/Healthy, 2=Student/Budget, 3=Family, 4=Workaholic.
P2 (Mood): 1=Happy/Energetic, 2=Tired/Lazy, 3=Sad/Stressed.
P5 (Income): 1=Low, 2=Medium, 3=High.

STRICT ACTION RULES (Output ONLY a single line, NO spaces after commas, NO conversational text):

1. If input is "shopping|X|CONTEXT_VECTOR" -> Generate exactly X random grocery items matching the profile.
   Category codes: 1=Meat, 2=Vegetables, 3=Dairy, 4=Drinks, 5=Bakery, 6=Sweets. Prices must scale with P5 (Income).
   Output format: CATEGORY_CODE:ITEM_NAME:UNIT_PRICE:PORTIONS:WEIGHT_GRAMS,...
   Example: 1:CHICKEN FILLET:3100:2:2434.5,2:TOMATOES:924:1:415.2

2. If input is "cooking|AVAILABLE_LIST|CONTEXT_VECTOR" -> Choose from numbered items to cook a dish.
   Output format: DISH_NAME:YIELD_PORTIONS:YIELD_WEIGHT|LOCAL_ID:USED_PORTIONS:USED_WEIGHT,...
   Example: CHICKEN SOUP WITH TOMATOES:4:2000|1:1:1200,2:2:800

3. If input is "consumption|AVAILABLE_FOOD|CONTEXT_VECTOR" -> Select food item by local ID and portion to eat.
   Output format: LOCAL_ID:PORTIONS:WEIGHT_GRAMS,...
   Example: 1:1:500,2:0.5:200

4. If input is "waste|SPOILED_LIST" -> Output item local ID to throw away and reason code (1=Mold, 2=Stale, 3=Expired, 4=Smell).
   Output format: LOCAL_ID:PORTIONS:WEIGHT:REASON_CODE,...
   Example: 1:1.0:400:1,2:0.5:200:2"""


class InferenceRequest(BaseModel):
    query: str

@app.post("/generate")
def generate_behavior(request: InferenceRequest):
    try:
        # Обращаемся к Ollama, установленной на хосте Ubuntu
        response = requests.post('http://ollama:11434/api/chat', json={
            "model": "qwen2.5:3b", 
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.query}
            ],
            "options": {
                "temperature": 0.8,
                "num_thread": 4,       # Строго 4 P-ядра тестового процессора i5-12450H
                "num_ctx": 4096,
                "num_predict": 800     # Запас, чтобы длинные списки продуктов не обрезались
            },
            "stream": False
        })
        result_text = response.json()['message']['content']
        return {"vector_output": result_text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama connection error: {str(e)}")
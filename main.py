import json
import os
import re
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

# Настраиваем структурированное логирование в stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ai-gateway")

app = FastAPI(title="Universal Stochastic AI Gateway")

# Вытаскиваем URL Ollama из env-переменных контейнера
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

class PromptRequest(BaseModel):
    model: str = "qwen2.5:3b"
    prompt: str
    stream: bool = False
    format: str = "json"

@app.post("/api/generate")
async def generate_receipt(req: PromptRequest):
    logger.info(f"Received prompt for model {req.model}")
    
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": req.model,
        "prompt": req.prompt,
        "stream": False,
        "format": "json"
    }

    try:
        # Шлем запрос во внутреннюю Ollama
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code != 200:
            logger.error(f"Ollama returned bad status: {resp.status_code}")
            return fallback_response("OLLAMA_BAD_STATUS")

        response_data = resp.json()
        raw_text = response_data.get("response", "").strip()

        # Очищаем markdown-грязь (убираем блоки ```json и ```)
        clean_json_text = re.sub(r"```json\s*|```", "", raw_text).strip()

        # Проверяем JSON на валидность перед отправкой в Go-воркер
        parsed_json = json.loads(clean_json_text)
        logger.info("Successfully validated LLM JSON response")
        return {"response": json.dumps(parsed_json, ensure_ascii=False)}

    except json.JSONDecodeError:
        logger.warning("LLM returned malformed JSON. Triggering secure fallback payload.")
        return fallback_response("MALFORMED_LLM_JSON")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to reach Ollama host: {e}")
        return fallback_response("OLLAMA_OFFLINE")

def fallback_response(reason: str):
    """Генератор безопасного чека-заглушки для предотвращения падения Go-воркера"""
    fallback = {
        "receipt_no": f"BILL-FALLBACK-{reason}-2026",
        "total_price": 490.0,
        "items": [
            {"item_id": "fallback-1", "name": "Продукты (Fallback)", "category": "Разное", "portions": 1.0, "weight_grams": 1000.0, "price": 490.0}
        ]
    }
    return {"response": json.dumps(fallback, ensure_ascii=False)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11435)

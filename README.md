# AI Stochastic Fridge Behavior Gateway (Python)

Dedicated, isolated inference microservice engineered for stochastic modeling of digital resident behavior within a smart fridge simulator. The system operates strictly as a deterministic, lightweight data refinery: it consumes structured contextual payload parameters from a remote Go simulator, computes human behavior patterns via a localized LLM, and returns dense, flat string vectors optimized for high-performance CPU execution without discrete GPU dependencies.

## Tech Stack
* **Language:** Python (3.11-slim)
* **API Framework:** FastAPI (Uvicorn ASGI Server)
* **Local Inference Engine:** Ollama API (Bare-metal CPU execution)
* **Language Model:** Qwen 2.5 (3B Parameters, GGUF Q4_K_M/Q5_K_M)
* **Containerization:** Docker & Docker Compose

## Architectural Flow & Background Workers
The microservice exposes a secure networking boundary, abstracting complex behavioral logic away from the primary execution core. To eliminate serialization overhead during the neural compute phase, the gateway compresses all pipeline interactions down to uniform token vectors:

1. **Inbound Parameter Decoupling:** FastAPI intercepts structurally validated Pydantic payloads containing separate fields for action, items, and context, protecting the 3B model from unparsed raw text pollution.
2. **Deterministic Schema Enforcement:** Requests pass down to Ollama under a balanced random temperature (`0.6`) to fuel living market behavior variety, but output structures are physically locked via a native `RESPONSE_SCHEMA` constraint to guarantee a raw JSON container with a single `vector` property.
3. **Aggressive CPU Latency Controls:** Custom, explicit keyword boundaries (`stop`) are injected directly into the execution configuration. The moment the model hits a newline sequence or attempts to mirror the input query, the host engine instantly halts processing on Intel Core i5 physical P-cores, dropping inference wait times to a stable **~1.2 – 3.5 seconds**.
4. **Flat Comma-Separated Extraction:** The gateway outputs a flat, dense string token block delimited solely by colons and commas, perfectly optimized for microsecond unpacking inside your remote Go streaming consumer.

## Topology & Infrastructure Routing

The service functions within an isolated Docker perimeter on your headless Ubuntu machine, exposing port 8000 externally while maintaining standard local loops for model interactions.

| Target Component | Connection String Endpoint | Attached Network | Purpose / Operational Scope |
| :--- | :--- | :--- | :--- |
| **fridge-ai-gateway** | `http://gateway:8000` | `bridge` (Docker Network) | FastAPI Gateway endpoint receiving structured JSON calls from the Go machine. |
| **fridge-ollama** | `http://ollama:11434` | `bridge` (Docker Network) | Bare-metal Ollama daemon running low-level matrix math restricted to 4 threads. |
| **model-puller** | Internal Curl Utility | `bridge` (Docker Network) | Asynchronous automatic bootstrap handler for the `qwen2.5:3b` image. |

### Network Segmentation Strategy
*   **AP Isolation Bypass Bridge:** Standard home wireless networks isolate client devices within the local boundary. External manual execution tests (Postman) and Go integration routines connect securely via an **SSH Tunnel (Port Forwarding)** spawned using the local flag `-L 8000:127.0.0.1:8000`.

## Environment Specifications
*   `action`: Inbound operational routing flag mapping to simulation subroutines (`shopping`, `cooking`, `consumption`, `waste`).
*   `items`: Dynamic content matrix tracking item limits or structured array indexes derived from active Redis memory states.
*   `context`: A compressed comma-separated string containing behavioral, environmental, and financial resident tags.
*   `vector_output`: Outbound flat string ledger destination channel for processed transaction charts.

## Deployment & Graceful Shutdown

Before triggering container build routines, ensure that the shared workspace containing your code files is synchronized and accessible:

```bash
docker compose up -d --build --force-recreate
```

To update the system configuration layers cleanly with an invalidated application cache after updating code methods or prompts:

```bash
docker compose down && docker compose up -d --build
```

### Lifecycle Guarantees (Hard Purge Routine)
To clean the development environment down to a blank state and reclaim server disc space, the engine explicitly drops mounted data boundaries, wiping out downloaded model weights and container volumes:

```bash
docker compose down -v && docker system prune -a --volumes -f
```

# Iraqi Support Ticket AI - Inference App

This app loads the Qwen2.5-3B-Instruct base model plus the trained LoRA adapter once at startup and exposes ticket inference through FastAPI.

## 1. Prepare environment

```bash
pip install -r requirements.txt
cp .env.example .env
```

Make sure `ADAPTER_PATH` points to the trained adapter directory (`adapter_config.json` and `adapter_model.safetensors`).

## 2. Start the API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Swagger UI is available at `/docs`.

## 3. Example request

```bash
curl -X POST http://127.0.0.1:8000/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"seq": 0, "text": "حولت 50 الف والمعاملة بعدها معلقة"},
      {"seq": 1, "text": "رقم العملية 928173"}
    ],
    "generate_report": true
  }'
```

The result contains category, priority, entities, confidence, per-category probabilities/logits, escalation state, department, and Arabic draft report.

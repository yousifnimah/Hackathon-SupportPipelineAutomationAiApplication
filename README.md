# Iraqi Support Ticket AI - Inference App

This app loads the Qwen2.5-3B-Instruct base model with a trained LoRA adapter fine-tuned on an Iraqi Arabic conversational support dataset, and exposes ticket inference through FastAPI.
## 1. Prepare environment

```bash
pip install -r requirements.txt
cp .env.example .env
```

Make sure `ADAPTER_PATH` points to the trained adapter directory (`adapter_config.json` and `adapter_model.safetensors`).

## 2. Start the API

```bash
uvicorn app:app --host 0.0.0.0 --port 8111
```

Swagger UI is available at `/docs`.

## 3. Example request

```bash
curl -X POST http://127.0.0.1:8111/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"seq": 0, "text": "حولت 50 الف والمعاملة بعدها معلقة"},
      {"seq": 1, "text": "رقم العملية 928173"}
    ],
    "generate_report": true
  }'
```

The result contains:

- **Category**
- **Priority**
- **Extracted entities**
- **Confidence score**
- **Per-category probabilities**
- **Per-category logits**
- **Escalation status**
- **Escalation reason**
- **Assigned department**
- **Arabic draft report**

### Example of Expected Response

```json
{
  "category": "account_issue",
  "priority": "medium",
  "entities": {
    "amount": null,
    "transaction_id": null,
    "recipient": null,
    "account_id": "550101",
    "date": null
  },
  "confidence": 0.9999,
  "category_probabilities": {
    "failed_transfer": 0.00003,
    "wrong_recipient": 0.000004,
    "payment_pending": 0.000006,
    "login_problem": 0.000018,
    "card_issue": 0.000008,
    "account_issue": 0.9999,
    "agent_dispute": 0.000016,
    "other": 0.000011
  },
  "category_logits": {
    "failed_transfer": 21.14,
    "wrong_recipient": 19.25,
    "payment_pending": 19.61,
    "login_problem": 20.73,
    "card_issue": 19.89,
    "account_issue": 31.64,
    "agent_dispute": 20.61,
    "other": 20.25
  },
  "escalated": false,
  "escalation_reason": null,
  "department": "Account",
  "draft_report": "ملخص الحالة:\nأبلغ العميل عن مشكلة في الحساب رقم 550101.\n\nالإجراء المقترح:\nمراجعة حالة الحساب من قبل قسم الحسابات والتحقق من سبب المشكلة."
}

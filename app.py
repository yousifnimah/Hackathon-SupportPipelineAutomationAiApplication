from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference import SupportTicketInference

engine: SupportTicketInference | None = None


class TicketMessage(BaseModel):
    seq: int = Field(ge=0)
    text: str = Field(min_length=1)


class InferenceRequest(BaseModel):
    messages: list[TicketMessage] = Field(min_length=1)
    generate_report: bool = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = SupportTicketInference()
    yield
    engine = None


app = FastAPI(
    title="Iraqi Support Ticket AI",
    version="1.0.0",
    lifespan=lifespan,
)

@app.post("/infer")
def infer(request: InferenceRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    try:
        messages = [message.model_dump() for message in request.messages]
        return engine.analyze_ticket(
            messages=messages,
            generate_report=request.generate_report,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

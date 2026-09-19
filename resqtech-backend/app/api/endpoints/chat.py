from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import current_user
from app.core.config import get_settings
from app.database.session import get_db
from app.models import ChatHistory
from app.schemas import ChatRequest, ChatResponse
from app.services import chat_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(body: ChatRequest, db: Session = Depends(get_db), user=Depends(current_user)):
    reply, answered_by = chat_service.ask(db, body.message)
    db.add(ChatHistory(user_id=user.id, session_id=body.session_id,
                       role="user", content=body.message))
    db.add(ChatHistory(user_id=user.id, session_id=body.session_id,
                       role="assistant", content=reply, answered_by=answered_by))
    db.commit()
    return ChatResponse(reply=reply, answered_by=answered_by,
                        data_mode=get_settings().data_mode,
                        sources=["risk_predictions", "alerts", "shelters",
                                 "resources", "roads", "hospitals"])

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import current_user
from app.core.security import create_access_token, verify_password
from app.database.session import get_db
from app.models import User
from app.schemas import LoginRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    if not user or not verify_password(body.password, user.password_hash):
        # Same message either way — do not leak which accounts exist.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Email or password is incorrect")
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "This account has been disabled")
    return TokenResponse(access_token=create_access_token(user.email, user.role),
                         role=user.role, full_name=user.full_name)


@router.get("/me")
def me(user: User = Depends(current_user)):
    return {"email": user.email, "full_name": user.full_name, "role": user.role}

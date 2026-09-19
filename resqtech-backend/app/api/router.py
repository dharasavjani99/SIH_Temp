from fastapi import APIRouter

from app.api.endpoints import (alerts, auth, chat, dashboard, impact, logistics,
                               reference, reports, resources, risk)

api_router = APIRouter(prefix="/api")
for module in (auth, dashboard, reference, risk, alerts, resources, logistics, impact,
               chat, reports):
    api_router.include_router(module.router)

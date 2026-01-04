from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
from app.service.chat_service import ChatService
from app.core.logger import get_logger
from app.core.metrics import ACTIVE_WS_CONNECTIONS

router = APIRouter()
logger = get_logger(__name__)


@router.websocket("/ws/chat")
async def chat_websocket(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket connected")
    ACTIVE_WS_CONNECTIONS.inc()

    try:
        while True:
            message = await ws.receive_text()
            data = json.loads(message)

            event_type = data.get("event_type")

            if event_type == "chat_request":
                payload = data.get("payload", {})
                await ChatService.handle_chat(ws, payload)

            else:
                await ws.send_json({
                    "event_type": "error",
                    "message": "Unsupported event type"
                })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")

    finally:
        ACTIVE_WS_CONNECTIONS.dec()

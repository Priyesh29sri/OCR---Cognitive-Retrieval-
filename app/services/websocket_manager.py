# app/services/websocket_manager.py
from fastapi.websockets import WebSocket
from loguru import logger

class WSConnectionManager:
    def __init__(self):
        # Keeps track of all active user connections
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("New WebSocket connection opened.")

    async def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info("WebSocket connection closed.")

    async def send_message(self, message: str, websocket: WebSocket):
        # Sends text data live to the client
        await websocket.send_text(message)

# Create a single global instance to reuse across the app
ws_manager = WSConnectionManager()

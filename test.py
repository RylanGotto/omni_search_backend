from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from schemas import ChatResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# resp = ChatResponse(sender="bot", message=token, type="stream")
import json


def receiverHandler(received_data):
    try:
        return ChatResponse(
            sender=received_data.get("sender"),
            message=received_data.get("message"),
            type=received_data.get("type"),
        )
    except Exception as e:
        print(e)


import asyncio


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    res = """
       Hello, I am a bot. I can help you with the following:
       Rap, talk shit
       Smoke fatties!
    """
    while True:
        data = await websocket.receive_text()
        received = receiverHandler(json.loads(data))
        print(received)
        for i in res:
            response = ChatResponse(sender="bot", message=i, type="stream")
            await asyncio.sleep(0.05)
            await websocket.send_json(response.dict())
        await websocket.send_json(
            ChatResponse(sender="bot", message="MSG_DONE", type="stream").dict()
        )

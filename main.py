import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from schemas import ChatResponse

from websockets.exceptions import ConnectionClosedOK
from query_data import get_agent

# Load the environment variables from the .env file
load_dotenv()

# Access the variables using os.environ
from fastapi.middleware.cors import CORSMiddleware

templates = Jinja2Templates(directory="templates")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/search")
async def test_chat(request: Request):
    return {"message": "Hello World"}


FORMAT = """
    FORMAT INSTRUCTIONS: DO NOT INCLUDE "Answer: " in your output.


"""
import json


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            if connection.client_state.value == 1001:
                self.disconnect(connection)
            if connection.client_state.value == 1:
                await connection.send_json(message)


manager = ConnectionManager()


@app.websocket("/ws/omnibot")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    agent = get_agent(manager)

    while True:
        # Receive and send back the client message
        user_msg = await websocket.receive_text()

        user_msg = json.loads(user_msg)
        # resp = ChatResponse(
        #     sender="user", message=user_msg.get("message"), type="user_message"
        # )
        # await manager.broadcast(resp.dict())

        # Construct a response
        # start_resp = ChatResponse(sender="bot", message="", type="start")
        # await manager.send_message(start_resp.dict(), websocket)

        # Send the message to the chain and feed the response back to the client
        await agent.arun(
            {"input": user_msg.get("message"), "format_instructions": FORMAT}
        )

        # Send the end-response back to the client
        # end_resp = ChatResponse(sender="bot", message="", type="end")
        # await manager.send_message(end_resp.dict(), websocket)

        # except WebSocketDisconnect as e:
        #     manager.disconnect(websocket)
        #     # TODO try to reconnect with back-off
        #     break
        # except ConnectionClosedOK:
        #     logging.info("ConnectionClosedOK")
        #     print(2)
        #     # TODO handle this?
        #     break
        # except Exception as e:
        #     logging.error(e)
        #     print("ERROR")
        #     resp = ChatResponse(
        #         sender="bot",
        #         message="Sorry, something went wrong. Try again.",
        #         type="error",
        #     )
        #     # await manager.send_message(resp.dict())

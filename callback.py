"""Callback handlers used in the app."""
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain.callbacks.base import AsyncCallbackHandler

from schemas import ChatResponse

from langchain.callbacks.base import BaseCallbackHandler

import asyncio
import json
import ast


class MyCustomHandler(AsyncCallbackHandler):
    def __init__(self, websocket) -> None:
        super().__init__()
        self.websocket = websocket
        self.output = {}

    async def on_tool_end(self, output: str, **kwargs):
        """Run when tool ends running."""

        output = ast.literal_eval(output)
        if not self.output:
            self.output = output
        await self.websocket.broadcast(self.output)


class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    def __init__(self, websocket) -> None:
        super().__init__()
        self.websocket = websocket
        self.has_seen_final_answer = False

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""

        if self.has_seen_final_answer:
            for i in token:
                resp = ChatResponse(sender="bot", message=i, type="stream")
                await asyncio.sleep(0.015)
                await self.websocket.broadcast(resp.dict())
        if "Final" in token:
            self.has_seen_final_answer = True

    async def on_llm_end(self, response, run_id, parent_run_id, **kwargs: Any) -> None:
        resp = ChatResponse(sender="bot", message="MSG_DONE", type="stream")
        await self.websocket.broadcast(resp.dict())

    def on_chat_model_start(self, serialized, messages, **kwargs: Any) -> None:
        self.has_seen_final_answer = False

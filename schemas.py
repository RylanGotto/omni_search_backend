"""Schemas for the chat app."""
from pydantic import BaseModel, validator


class ChatResponse(BaseModel):
    """Chat response schema."""

    sender: str
    message: str
    type: str

    @validator("sender")
    def sender_must_be_bot_or_you(cls, v):
        if v not in ["bot", "user"]:
            raise ValueError("sender must be bot or user")
        return v

    @validator("type")
    def validate_message_type(cls, v):
        if v not in ["start", "stream", "end", "error", "info", "user_message"]:
            raise ValueError("type must be start, stream or end")
        return v

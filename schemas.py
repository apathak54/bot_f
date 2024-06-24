"""Schemas for the chat app."""
from pydantic import BaseModel, validator


class ChatResponse(BaseModel):
    """Chat response schema."""

    sender: str
    message: str
    integer: int
    type: str

    @validator("sender")
    def sender_must_be_bot_or_you(cls, v):
        if v not in ["bot", "human"]:
            raise ValueError("sender must be bot or human")
        return v

    @validator("type")
    def validate_message_type(cls, v):
        if v not in ["start", "stream", "end", "error", "info","audio_stream",
                     "audio_start","audio_end"]:
            raise ValueError("type must be start, stream or end")
        return v
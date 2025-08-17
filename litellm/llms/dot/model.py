"""Dot Model"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ContentItem(BaseModel):
    id: int = Field(default=0)
    role: Optional[str] = Field(default=None)
    text: Optional[str] = Field(default=None)
    type: Optional[str] = Field(default=None)


class Usage(BaseModel):
    output_tokens: int
    prompt_tokens: int
    total_tokens: int


class Additional(BaseModel):
    finish_reason: str


class DotResponse(BaseModel):

    content: List[ContentItem]
    additional_data_messages: Additional
    usage: Usage
    model: str

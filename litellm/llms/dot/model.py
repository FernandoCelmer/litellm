"""Dot Model"""

from typing import List

from pydantic import BaseModel


class ContentItem(BaseModel):
    id: int
    role: str
    text: str
    type: str


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

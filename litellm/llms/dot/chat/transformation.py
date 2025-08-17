"""Dot Chat Config"""

from typing import List

from litellm.types.llms.openai import AllMessageValues
from litellm.llms.openai.openai import OpenAIConfig


class DotChatConfig(OpenAIConfig):

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str
    ) -> List[AllMessageValues]:
        
        messages = [
            {**message, "content": [{"type": "text", "text": message["content"]}]}
            for message in messages
        ]

        return super()._transform_messages(messages=messages, model=model)

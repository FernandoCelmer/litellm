"""Dot Chat Completion"""

from typing import (
    Optional,
    Union,
    cast
)

import httpx
import openai

import litellm
from openai import AsyncOpenAI

from litellm.types.utils import ModelResponse
from litellm.llms.openai.openai import OpenAIChatCompletion
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.dot.model import DotResponse
from litellm.llms.openai.common_utils import (
    OpenAIError,
    drop_params_from_unprocessable_entity_error,
)


class DotChatCompletion(OpenAIChatCompletion):

    def _convert_to_model_response_object(*_args, **kwargs):
            response_wrapper: DotResponse = DotResponse(**kwargs["response_object"])
            model_response_object: ModelResponse = kwargs["model_response_object"]

            model_response_object.usage.total_tokens = response_wrapper.usage.total_tokens
            model_response_object.usage.prompt_tokens = response_wrapper.usage.prompt_tokens

            for index, choice in enumerate(model_response_object.choices):
                choice.index = response_wrapper.content[index].id
                choice.message.role = response_wrapper.content[index].role
                choice.message.content = response_wrapper.content[index].text
                choice.finish_reason = response_wrapper.additional_data_messages.finish_reason

            return model_response_object

    async def acompletion(
        self,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        provider_config: BaseConfig,
        model: str,
        model_response: ModelResponse,
        logging_obj: Logging,
        timeout: Union[float, httpx.Timeout],
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        organization: Optional[str] = None,
        client=None,
        max_retries=None,
        headers=None,
        drop_params: Optional[bool] = None,
        stream_options: Optional[dict] = None,
        fake_stream: bool = False,
    ):
        response = None
        data = await provider_config.async_transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers or {},
        )
        for _ in range(
            2
        ):
            try:
                openai_aclient: AsyncOpenAI = self._get_openai_client(
                    is_async=True,
                    api_key=api_key,
                    api_base=api_base,
                    api_version=api_version,
                    timeout=timeout,
                    max_retries=max_retries,
                    organization=organization,
                    client=client,
                )

                logging_obj.pre_call(
                    input=data["messages"],
                    api_key=openai_aclient.api_key,
                    additional_args={
                        "headers": {
                            "Authorization": f"Bearer {openai_aclient.api_key}"
                        },
                        "api_base": openai_aclient._base_url._uri_reference,
                        "acompletion": True,
                        "complete_input_dict": data,
                    },
                )

                headers, response = await self.make_openai_chat_completion_request(
                    openai_aclient=openai_aclient,
                    data=data,
                    timeout=timeout,
                    logging_obj=logging_obj,
                )
                stringified_response = response.model_dump()

                logging_obj.post_call(
                    input=data["messages"],
                    api_key=api_key,
                    original_response=stringified_response,
                    additional_args={"complete_input_dict": data},
                )
                logging_obj.model_call_details["response_headers"] = headers
                final_response_obj = self._convert_to_model_response_object(
                    response_object=stringified_response,
                    model_response_object=model_response,
                    hidden_params={"headers": headers},
                    _response_headers=headers,
                )

                if fake_stream is True:
                    return self.mock_streaming(
                        response=cast(ModelResponse, final_response_obj),
                        logging_obj=logging_obj,
                        model=model,
                        stream_options=stream_options,
                    )

                return final_response_obj
            except openai.UnprocessableEntityError as e:
                if litellm.drop_params is True or drop_params is True:
                    data = drop_params_from_unprocessable_entity_error(e, data)
                else:
                    raise e
            except Exception as e:
                exception_response = getattr(e, "response", None)
                status_code = getattr(e, "status_code", 500)
                exception_body = getattr(e, "body", None)
                error_headers = getattr(e, "headers", None)
                if error_headers is None and exception_response:
                    error_headers = getattr(exception_response, "headers", None)
                message = getattr(e, "message", str(e))

                raise OpenAIError(
                    status_code=status_code,
                    message=message,
                    headers=error_headers,
                    body=exception_body,
                )

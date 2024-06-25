from typing import Any, Callable, cast, Tuple, TypeVar

import logging

from . import config_options
from modules.objects import FullArticle

from openai import OpenAIError, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
import tiktoken

from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

openai_client = OpenAI(api_key=config_options.OPENAI_KEY)

logger = logging.getLogger("osinter")


StrTuple = TypeVar("StrTuple", bound=Tuple[str, ...])


def extract_labeled(text: str, extraction_labels: StrTuple) -> StrTuple:
    """Given extraction_labels=[label_1, label_2, label_3]
    text should be formatted as:

    label_1: [A string of given length]
    label_2: [A string of given length]
    label_3: [A string of given length]
    """

    first_label = extraction_labels[0]
    unprocesseced_response = text.split(f"{first_label}:")[-1]

    extractions: list[str] = []

    for label in extraction_labels[1:]:
        response_parts = unprocesseced_response.split(f"{label}:")
        extractions.append(response_parts[0].strip())
        unprocesseced_response = response_parts[-1]

    extractions.append(unprocesseced_response)

    if len(extractions) != len(extraction_labels):
        raise IndexError("Not all labels are present")

    return cast(StrTuple, tuple(extractions))


def query_openai(prompts: list[ChatCompletionMessageParam]) -> str:
    @retry(
        wait=wait_random_exponential(min=1, max=3600),
        stop=stop_after_attempt(20),
        retry=retry_if_exception_type(OpenAIError),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
    )
    def query(q: list[ChatCompletionMessageParam]) -> ChatCompletion:
        return openai_client.chat.completions.create(
            model=config_options.OPENAI_MODEL,
            messages=q,
            n=1,
            temperature=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

    response = query(prompts).choices[0].message.content
    if isinstance(response, str):
        return response
    else:
        raise TypeError("OpenAI response was empty")


def query_and_extract(
    prompts: list[ChatCompletionMessageParam], labels: StrTuple
) -> None | StrTuple:
    @retry(
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((IndexError, TypeError)),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
    )
    def attempt() -> StrTuple:
        openai_response = query_openai(prompts)

        return extract_labeled(openai_response, labels)

    try:
        return attempt()
    except:
        logger.error(
            f'Unable to extract details from OpenAI response with labels "{" | ".join(labels)}"'
        )
        return None


def construct_description_prompts(
    texts: list[str], description_prompt: str, instruction_prompt: str
) -> list[ChatCompletionMessageParam]:
    enc = tiktoken.encoding_for_model(config_options.OPENAI_MODEL)

    openai_messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": description_prompt},
        {"role": "user", "content": ""},
    ]

    prompt_length = len(enc.encode(description_prompt + instruction_prompt))

    for text in texts:
        prompt_length += len(enc.encode(text))

        # Using 80% of available tokens to leave room for answer
        if prompt_length > (config_options.OPENAI_TOKEN_LIMIT * 0.8):
            break

        openai_messages[-1]["content"] += f'"""{text}"""'  # type: ignore

    openai_messages.append({"role": "user", "content": instruction_prompt})

    return openai_messages

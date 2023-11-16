from typing import Any, cast, Tuple, TypeVar

import logging

from . import config_options

from openai import OpenAIError, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

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
    unprocesseced_response = text.split(first_label)[-1]

    extractions: list[str] = []

    for label in extraction_labels[1:]:
        response_parts = unprocesseced_response.split(label)
        extractions.append(response_parts[0])
        unprocesseced_response = response_parts[-1]

    extractions.append(unprocesseced_response)

    return cast(StrTuple, tuple(extractions))


def query_openai(prompts: list[ChatCompletionMessageParam]) -> str | None:
    @retry(
        wait=wait_random_exponential(min=1, max=600),
        stop=stop_after_attempt(10),
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

    try:
        return query(prompts).choices[0].message.content
    except RetryError:
        return None


def construct_description_prompts(
    keywords: list[str], representative_docs: list[str]
) -> tuple[list[ChatCompletionMessageParam], tuple[str, str, str]]:
    descriptionPrompt = f"""
I have topic is described by the following keywords: {' '.join(keywords)}
The following documents delimited by triple quotes are a small but representative subset of all documents in the topic:
"""
    instructionPrompt = """
Based on the information above return the following

A title for this topic of at most 10 words
A description of the topic with a length of 1 to 2 sentences
A summary of the topic with a length of 4 to 6 sentences

The returned information should be in the following format:
topic_title: <title>
topic_description: <description>
topic_summary: <summary>
    """

    openai_messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": descriptionPrompt}
    ]
    openai_messages.extend(
        [
            {"role": "user", "content": f'"""{doc[:3200]}"""'}
            for doc in representative_docs
        ]
    )
    openai_messages.append({"role": "user", "content": instructionPrompt})

    return openai_messages, ("topic_title: ", "topic_description: ", "topic_summary: ")

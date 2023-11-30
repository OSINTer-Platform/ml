from typing import Any, cast, Tuple, TypeVar

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
    articles: list[FullArticle],
) -> tuple[list[ChatCompletionMessageParam], tuple[str, str, str]]:
    description_prompt = f"""
I have topic containing a set of news articles, descriping a topic within cybersecurity.
The following documents delimited by triple quotes are the title and description of a small but representative subset of all documents in the topic.
As such you should choose the broadest description of the topic that fits the articles:
"""
    instruction_prompt = """
Based on the information above return the following

A title for this topic of at most 10 words
A description of the topic with a length of 1 to 2 sentences
A summary of the topic with a length of 4 to 6 sentences

The returned information should be in the following format:
topic_title: <title>
topic_description: <description>
topic_summary: <summary>
    """

    enc = tiktoken.encoding_for_model(config_options.OPENAI_MODEL)

    openai_messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": description_prompt},
        {"role": "user", "content": ""},
    ]

    prompt_length = len(enc.encode(description_prompt + instruction_prompt))

    for article in articles:
        article_description = article.title + " " + article.description
        prompt_length += len(enc.encode(article_description))

        # Using 80% of available tokens to leave room for answer
        if prompt_length > (config_options.OPENAI_TOKEN_LIMIT * 0.8):
            break

        openai_messages[-1]["content"] += f'"""{article_description}"""'  # type: ignore

    openai_messages.append({"role": "user", "content": instruction_prompt})

    return openai_messages, ("topic_title: ", "topic_description: ", "topic_summary: ")

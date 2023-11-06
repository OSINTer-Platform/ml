from typing import Any, Callable, cast
from nptyping import NDArray

import logging
from hashlib import md5

from tenacity import RetryError, before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from modules.objects import FullArticle, FullCluster
from .inference import OpenAIMessage, extract_labeled, query_openai, construct_description_prompts

from .multithreading import process_threaded
from . import embedding_model

from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger("osinter")

umap_model = UMAP(
    n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42
)
hdbscan_model = HDBSCAN(
    min_cluster_size=10,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)
vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))


def describe_cluster(
    keywords: list[str], representative_docs: list[str], use_openai: bool = True
) -> tuple[str, str, str]:

    @retry(
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(IndexError),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
    )
    def get_open_ai_description(prompts: list[OpenAIMessage], labels: tuple[str, str, str]):

        openai_response = query_openai(prompts)

        if openai_response is None:
            logger.warn(f'Response for cluster with title "{title}" failed to query openai')
            return None

        return extract_labeled(openai_response, labels)

    # Defaults in case openai return nonsensical answer
    title = " | ".join([keyword.capitalize() for keyword in keywords])
    description = " | ".join([keyword.capitalize() for keyword in keywords])
    summary = ""
    default_response = title, description, summary

    if not use_openai:
        return default_response

    prompts, labels = construct_description_prompts(keywords, representative_docs)

    try:
        response = get_open_ai_description(prompts, labels)

        if response:
            return response
        else:
            return default_response

    except RetryError:
        logger.error(
            f'Unable to extract details from OpenAI response with labels "{" | ".join(labels)}"'
        )
        return default_response


def fit_topic_model(
    articles: list[FullArticle], embeddings: NDArray[Any, Any]
) -> BERTopic:
    topic_model = BERTopic(
        # Pipeline models
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        # Hyperparameters
        top_n_words=10,
        verbose=True,
    )

    contents = [article.content for article in articles]

    logger.debug("Fitting to BERTopic")
    topic_model.fit_transform(contents, embeddings)
    return topic_model


def create_clusters(
    articles: list[FullArticle],
    embeddings: NDArray[Any, Any],
    save_model: None | str,
    use_openai: bool = True,
) -> list[FullCluster]:
    def create_cluster(record: dict[str, Any], topic_numbers: list[int]) -> FullCluster:
        topic_docs: list[FullArticle] = []
        representative_docs: list[FullArticle] = []

        for topic, article in zip(topic_numbers, articles):
            if topic == record["Topic"]:
                topic_docs.append(article)

            if article.content in record["Representative_Docs"]:
                representative_docs.append(article)

        title, description, summary = describe_cluster(
            record["Representation"], record["Representative_Docs"], use_openai
        )

        return FullCluster(
            id=md5(str(record["Topic"]).encode("utf-8")).hexdigest(),
            nr=record["Topic"],
            document_count=record["Count"],
            title=title,
            description=description,
            summary=summary,
            keywords=record["Representation"],
            representative_documents=[article.id for article in representative_docs],
            documents={article.id for article in topic_docs},
            dating={article.publish_date for article in topic_docs},
        )

    logger.debug("Fitting new model")
    topic_model = fit_topic_model(articles, embeddings)

    if save_model:
        logger.debug(f'Saving model to file "{save_model}"')
        topic_model.save(save_model, serialization="pickle")

    logger.debug("Normalizing and enriching cluster data")

    records = topic_model.get_topic_info().to_dict("records")

    # TODO: Check type of returned topics
    topic_numbers = cast(list[int], topic_model.topics_)

    for topic, article in zip(topic_numbers, articles):
        article.ml["cluster"] = topic

    handle_record: Callable[
        [dict[str, Any]], FullCluster
    ] = lambda record: create_cluster(record, topic_numbers)
    clusters: list[FullCluster] = process_threaded(records, handle_record)

    return clusters


def cluster_new_articles(
    articles: list[FullArticle],
    embeddings: NDArray[Any, Any],
    clusters: list[FullCluster],
    saved_model: str,
) -> None:
    def update_clusters(
        clusters: list[FullCluster], articles: list[FullArticle], topic_numbers: list[int]
    ):
        for cluster in clusters:
            if cluster.nr not in topic_numbers:
                continue

            for topic, article in zip(topic_numbers, articles):
                if topic == cluster.nr:
                    cluster.documents.add(article.id)
                    cluster.dating.add(article.publish_date)

                    cluster.document_count = len(cluster.documents)

    logger.debug(f'Loading model from file "{saved_model}"')

    topic_model = BERTopic.load(saved_model)
    topic_numbers, _ = topic_model.transform(
        [article.content for article in articles], embeddings
    )

    # Despite typed differently, topic_numbers is numpy int64 and needs conversion
    topic_numbers = [int(nr) for nr in topic_numbers]

    for topic, article in zip(topic_numbers, articles):
        article.ml["cluster"] = topic

    update_clusters(clusters, articles, topic_numbers)

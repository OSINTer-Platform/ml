import pickle
from typing import Any, Callable
from nptyping import NDArray

import logging
from hashlib import md5
from openai.types.chat import ChatCompletionMessageParam

from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
)

from modules.objects import FullArticle, FullCluster

from .multithreading import process_threaded
from .inference import (
    extract_labeled,
    query_openai,
    construct_description_prompts,
)

from umap import UMAP
from hdbscan import HDBSCAN
from hdbscan.flat import HDBSCAN_flat, approximate_predict_flat

logger = logging.getLogger("osinter")

UMAP_MODEL_PATH = "./models/cluster_umap"
HDBSCAN_MODEL_PATH = "./models/cluster_hdbscan"
HDBSCAN_EPSILON = 0.2


def describe_cluster(
    cluster_nr: int, articles: list[FullArticle]
) -> tuple[str, str, str]:
    @retry(
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(IndexError),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
    )
    def get_open_ai_description(
        prompts: list[ChatCompletionMessageParam], labels: tuple[str, str, str]
    ) -> None | tuple[str, str, str]:
        openai_response = query_openai(prompts)

        if openai_response is None:
            logger.warn(
                f'Response for cluster with title "{title}" failed to query openai'
            )
            return None

        return extract_labeled(openai_response, labels)

    # Defaults in case openai return nonsensical answer
    title = f"Cluster {cluster_nr}"
    description = f"This cluster contains {len(articles)} articles"
    summary = "A summary isn't available"
    default_response = title, description, summary

    prompts, labels = construct_description_prompts(articles)

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


def create_cluster(
    cluster_nr: int, articles: list[FullArticle], labels: list[int]
) -> FullCluster:
    relevant: list[FullArticle] = []
    not_relevant: list[FullArticle] = []

    for i, article in enumerate(articles):
        if labels[i] == cluster_nr:
            relevant.append(article)
        else:
            not_relevant.append(article)

    title, description, summary = describe_cluster(cluster_nr, relevant)
    keywords: list[str] = []

    return FullCluster(
        id=md5(str(cluster_nr).encode("utf-8")).hexdigest(),
        nr=cluster_nr,
        document_count=len(relevant),
        title=title,
        description=description,
        summary=summary,
        keywords=keywords,
        documents={article.id for article in relevant},
        dating={article.publish_date for article in relevant},
    )


def update_articles(
    articles: list[FullArticle], clusters: list[FullCluster], labels: list[int]
) -> None:
    logger.debug("Modifying articles")

    cluster_lookup = {cluster.nr: cluster for cluster in clusters}

    for i, article in enumerate(articles):
        current_label = labels[i]
        if current_label in cluster_lookup:
            article.ml.cluster = cluster_lookup[current_label].id
        else:
            logger.error(
                f'Missing cluster description for cluster {current_label} for article: "{article.id}: {article.title}"'
            )


def create_clusters(
    articles: list[FullArticle],
    embeddings: NDArray[Any, Any],
) -> list[FullCluster]:
    logger.debug("Fitting new UMAP model")
    umap = UMAP(
        min_dist=0, n_neighbors=7, n_components=20, metric="cosine", random_state=42
    )
    reduced_embeddings = umap.fit_transform(embeddings)

    logger.debug(f'Saving UMAP model to file "{UMAP_MODEL_PATH}"')
    with open(UMAP_MODEL_PATH, "wb") as f:
        pickle.dump(umap, f)

    logger.debug("Fitting new HDBSCAN model")
    hdbscan = HDBSCAN_flat(
        reduced_embeddings,
        min_cluster_size=5,
        min_samples=5,
        cluster_selection_epsilon=HDBSCAN_EPSILON,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = hdbscan.labels_

    logger.debug(f'Saving HDBSCAN model to file "{HDBSCAN_MODEL_PATH}"')
    with open(HDBSCAN_MODEL_PATH, "wb") as f:
        pickle.dump(hdbscan, f)

    norm_labels = [int(label) for label in labels]

    logger.debug(f"Creating {max(norm_labels)} clusters")
    cluster_nrs = list(set(norm_labels))

    handle_cluster: Callable[[int], FullCluster] = lambda cluster_nr: create_cluster(
        cluster_nr, articles, norm_labels
    )

    clusters: list[FullCluster] = process_threaded(cluster_nrs, handle_cluster)

    update_articles(articles, clusters, norm_labels)

    return clusters


def cluster_new_articles(
    articles: list[FullArticle],
    embeddings: NDArray[Any, Any],
    clusters: list[FullCluster],
) -> None:
    def update_clusters(
        clusters: list[FullCluster],
        articles: list[FullArticle],
    ) -> None:
        for cluster in clusters:
            relevant: list[FullArticle] = [
                article for article in articles if article.ml.cluster == cluster.id
            ]

            cluster.documents = {article.id for article in relevant}
            cluster.dating = {article.publish_date for article in relevant}

            cluster.document_count = len(cluster.documents)

    logger.debug(f'Loading UMAP from file "{UMAP_MODEL_PATH}"')
    with open(UMAP_MODEL_PATH, "rb") as f:
        umap: UMAP = pickle.load(f)

    logger.debug(f'Loading HDBSCAN from file "{HDBSCAN_MODEL_PATH}"')
    with open(HDBSCAN_MODEL_PATH, "rb") as f:
        hdbscan: HDBSCAN = pickle.load(f)

    reduced_embeddings = umap.transform(embeddings)
    labels = approximate_predict_flat(
        hdbscan, reduced_embeddings, cluster_selection_epsilon=HDBSCAN_EPSILON
    )[0]
    norm_labels = [int(label) for label in labels]

    update_articles(articles, clusters, norm_labels)
    update_clusters(clusters, articles)

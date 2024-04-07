import pickle
from typing import Any, Callable
from nptyping import NDArray

import logging
from hashlib import md5

from modules.objects import FullArticle, FullCluster

from .multithreading import process_threaded
from .inference import (
    query_and_extract,
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
    labels = ("topic_title", "topic_description", "topic_summary")

    prompts = construct_description_prompts(
        [a.title + " " + a.description for a in articles],
        """I have topic containing a set of news articles, descriping a topic within cybersecurity.
The following documents delimited by triple quotes are the title and description of a small but representative subset of all documents in the topic.
As such you should choose the broadest description of the topic that fits the articles:""",
        """Based on the information above return the following
A title for this topic of at most 10 words
A description of the topic with a length of 1 to 2 sentences
A summary of the topic with a length of 4 to 6 sentences

The returned information should be in the following format:
topic_title: <title>
topic_description: <description>
topic_summary: <summary>
""",
    )

    response = query_and_extract(prompts, labels)

    if response:
        return response
    else:
        logger.error(f"Unable to generate descriptions for cluster {cluster_nr}")
        return (
            f"Cluster {cluster_nr}",
            f"This cluster contains {len(articles)} articles",
            "A summary isn't available",
        )


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

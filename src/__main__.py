import logging
import os
import pickle
from typing import Any, cast

from nptyping import NDArray
from umap import UMAP
from modules.objects import FullCluster, FullArticle

from src.cluster import create_clusters, cluster_new_articles
from src.inference import OpenAIMessage, query_openai
from src.multithreading import process_threaded

from .map import calc_cords, calc_similar

from . import config_options, embedding_model
from modules.elastic import ArticleSearchQuery

logger = logging.getLogger("osinter")


def calc_embeddings(articles: list[FullArticle]) -> NDArray[Any, Any]:
    logger.debug("Pre-calculating embeddings for articles")
    contents = [article.content for article in articles]

    return cast(
        NDArray[Any, Any],
        embedding_model.encode(contents, show_progress_bar=True, convert_to_numpy=True),
    )


def get_documents() -> tuple[list[FullArticle], list[FullCluster]]:
    logger.info("Downloading articles")
    articles = config_options.es_article_client.query_documents(
        ArticleSearchQuery(limit=0), True
    )[0]

    logger.info("Downloading clusters")
    clusters = config_options.es_cluster_client.query_all_documents()

    return articles, clusters


def update_articles() -> None:
    logger.info("Verifying presence of topic and umap models")
    for model in ["topic_model", "umap"]:
        if not os.path.exists(f"./models/{model}"):
            logger.error(f"Missing {model} model")
            raise RuntimeError(f"Model at ./models/{model} not found")

    articles, clusters = get_documents()
    embeddings = calc_embeddings(articles)

    logger.info("Loading UMAP model")

    with open("./models/umap", "rb") as f:
        umap: UMAP = pickle.load(f)

    logger.info("Calculating cords")
    calc_cords(articles, embeddings, umap)

    logger.info("Calculating similar articles")
    calc_similar(articles, 20)

    logger.info("Clustering articles")
    cluster_new_articles(
        articles,
        embeddings,
        clusters,
        saved_model="./models/topic_model",
    )

    logger.info(f"Updating {len(articles)} articles")

    updated_articles_count = config_options.es_article_client.update_documents(
        articles, ["ml", "similar"]
    )

    logger.info(f"Updated {updated_articles_count} articles")
    logger.info(f"Updating {len(clusters)} clusters")

    updated_clusters_count = config_options.es_cluster_client.update_documents(
        clusters, ["document_count", "documents", "dating"]
    )

    logger.info(f"Updated {updated_clusters_count} clusters")


def create_topic_model() -> None:
    articles, old_clusters = get_documents()
    embeddings = calc_embeddings(articles)

    logger.info("Generating clusters and topic model. This could take some time")
    new_clusters = create_clusters(
        articles, embeddings, "./models/topic_model", bool(config_options.OPENAI_KEY)
    )

    old_cluster_ids = [cluster.id for cluster in old_clusters]
    new_cluster_ids = [cluster.id for cluster in new_clusters]
    clusters_to_remove = {id for id in old_cluster_ids if id not in new_cluster_ids}

    logger.info(f"Generated {len(new_clusters)} clusters. Saving them")
    saved_clusters = config_options.es_cluster_client.save_documents(new_clusters, True)
    logger.info(f"Save {saved_clusters} clusters")

    logger.info(f"Removing {len(clusters_to_remove)} old clusters")
    removed_clusters = config_options.es_cluster_client.delete_document(
        clusters_to_remove
    )
    logger.info(f"Removed {removed_clusters} old clusters")

    logger.info("Creating umap model")
    model = calc_cords(articles, embeddings)

    logger.info("Saving umap model")

    with open("./models/umap", "wb") as f:
        pickle.dump(model, f)

    logger.info("Calculating similar articles")
    calc_similar(articles, 20)

    logger.info(f"Updating {len(articles)} articles")

    updated_articles_count = config_options.es_article_client.update_documents(
        articles, ["ml", "similar"]
    )

    logger.info(f"Updated {updated_articles_count} articles")


def summarize_articles() -> None:
    def get_prompt(content: str) -> list[OpenAIMessage]:
        return [
            {
                "role": "system",
                "content": "Please summarize the following article delimited by triple quotes, in 3-6 sentences and in a way that preserves the technical details. Assume the audience is experts in the field.",
            },
            {"role": "user", "content": f'"""{content[:10_000]}"""'},
        ]

    def summarize_article(article: FullArticle) -> None:
        if article.summary:
            return

        prompt = get_prompt(article.content)
        article.summary = query_openai(prompt)

    logger.info(f"Downloading articles")
    articles = config_options.es_article_client.query_documents(
        ArticleSearchQuery(limit=0, sort_order="desc", sort_by="publish_date"), True
    )[0]

    logger.info(f"Starting summarization")
    try:
        process_threaded(articles, summarize_article)
        logger.info("Summarization done.")
    except:
        logger.error("Summarization failed! Saving processed information and exiting")

    logger.info(f"Updating {len(articles)} articles")

    updated_articles_count = config_options.es_article_client.update_documents(
        articles, ["summary"]
    )

    logger.info(f"Updated {updated_articles_count} articles")



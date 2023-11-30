import logging
import os
from typing import Any, cast
from openai.types.chat import ChatCompletionMessageParam

import typer
from nptyping import NDArray
from sentence_transformers import SentenceTransformer
from src.cluster import create_clusters, cluster_new_articles
from src.inference import query_openai
from src.multithreading import process_threaded

from .map import calc_cords, calc_similar

from . import config_options
from modules.elastic import ArticleSearchQuery
from modules.objects import FullCluster, FullArticle

logger = logging.getLogger("osinter")
app = typer.Typer(no_args_is_help=True)


def calc_embeddings(articles: list[FullArticle]) -> NDArray[Any, Any]:
    logger.debug("Loading embedding model")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

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


@app.command()
def update_articles() -> None:
    logger.info("Verifying presence of topic and umap models")
    for model in ["topic_model", "umap"]:
        if not os.path.exists(f"./models/{model}"):
            logger.error(f"Missing {model} model")
            raise RuntimeError(f"Model at ./models/{model} not found")

    articles, clusters = get_documents()
    embeddings = calc_embeddings(articles)

    logger.info("Calculating cords")
    calc_cords(articles, embeddings, False)

    logger.info("Calculating similar articles")
    calc_similar(articles, 20)

    logger.info("Clustering articles")
    cluster_new_articles(
        articles,
        embeddings,
        clusters,
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




@app.command()
def create_models() -> None:
    if not bool(config_options.OPENAI_KEY):
        raise Exception("OpenAI needed to generate clusters")

    articles, old_clusters = get_documents()
    embeddings = calc_embeddings(articles)

    logger.info("Generating clusters and topic model. This could take some time")
    new_clusters = create_clusters(articles, embeddings)

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
    calc_cords(articles, embeddings, True)

    logger.info("Calculating similar articles")
    calc_similar(articles, 20)

    logger.info(f"Updating {len(articles)} articles")

    updated_articles_count = config_options.es_article_client.update_documents(
        articles, ["ml", "similar"]
    )

    logger.info(f"Updated {updated_articles_count} articles")


@app.command()
def summarize_articles(all: bool = False, batch_size: int = 100) -> None:
    def get_prompt(content: str) -> list[ChatCompletionMessageParam]:
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
        ArticleSearchQuery(
            limit=0 if all else 200, sort_order="desc", sort_by="publish_date"
        ),
        True,
    )[0]

    articles_without_summary = [article for article in articles if not article.summary]
    updated_articles_count = 0

    for i, article_sublist in enumerate(
        [
            articles_without_summary[i : i + batch_size]
            for i in range(0, len(articles_without_summary), batch_size)
        ]
    ):
        logger.info(
            f"Starting summarization of batch {i} out of {int(len(articles_without_summary) / batch_size) + 1}"
        )
        try:
            process_threaded(article_sublist, summarize_article)
            logger.info("Summarization done.")
        except:
            logger.exception(
                f"Summarization failed! Saving processed information and exiting"
            )
            break
        finally:
            articles_to_update = [
                article for article in article_sublist if article.summary
            ]

            logger.info(f"Updating {len(articles_to_update)} articles of batch {i}")

            updated_articles_count += config_options.es_article_client.update_documents(
                articles_to_update, ["summary"]
            )

    logger.info(f"Updated {updated_articles_count} articles")


if __name__ == "__main__":
    app()

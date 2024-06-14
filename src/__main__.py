import logging
import os
from typing import Any, cast
from openai.types.chat import ChatCompletionMessageParam
from datetime import datetime

import typer
from nptyping import NDArray
from sentence_transformers import SentenceTransformer

from modules.objects.articles import BaseArticle
from modules.objects.cves import FullCVE
from src.cluster import (
    create_clusters,
    cluster_new_articles,
    UMAP_MODEL_PATH as CLUSTER_UMAP_PATH,
    HDBSCAN_MODEL_PATH as CLUSTER_HDBSCAN_PATH,
)
from src.cve import (
    generate_cve_title,
    get_article_common_keywords,
    query_nvd,
    sort_articles_by_cves,
    validate_cve,
)
from src.inference import query_openai
from src.multithreading import process_threaded
from src.map import calc_cords, calc_similar, UMAP_MODEL_PATH as MAP_UMAP_PATH

from . import config_options
from modules.elastic import ArticleSearchQuery, CVESearchQuery
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
    for model_path in [MAP_UMAP_PATH, CLUSTER_UMAP_PATH, CLUSTER_HDBSCAN_PATH]:
        if not os.path.exists(model_path):
            logger.error(f"Missing {model_path} model")
            raise RuntimeError(f"Model at {model_path} not found")

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
    saved_clusters = config_options.es_cluster_client.save_documents(new_clusters)
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
    if not bool(config_options.OPENAI_KEY):
        raise Exception("OpenAI needed to generate clusters")

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


def _create_cves(
    start_date: datetime | None = None,
    sorted_articles: dict[str, list[BaseArticle]] | None = None,
) -> None:
    if sorted_articles is None:
        logger.info("Downloading articles")
        articles = config_options.es_article_client.query_documents(
            ArticleSearchQuery(limit=0), False
        )[0]

        sorted_articles = sort_articles_by_cves(articles)

    for raw_cves in query_nvd(start_date):
        cves: list[FullCVE] = []

        logger.info("Validating CVEs")
        for raw_cve in raw_cves:
            articles = []
            if raw_cve["cve"]["id"] in sorted_articles:
                articles = sorted_articles[raw_cve["cve"]["id"]]

            cves.append(validate_cve(raw_cve["cve"], articles))

        logger.info("Creating CVE titles")

        cves = process_threaded(cves, generate_cve_title, 32)

        logger.info("Saving CVEs")
        config_options.es_cve_client.save_documents(cves, chunk_size=500)


@app.command()
def create_cves() -> None:
    _create_cves()


@app.command()
def update_cves() -> None:
    logger.info("Downloading articles")
    articles = config_options.es_article_client.query_documents(
        ArticleSearchQuery(limit=0), False
    )[0]
    sorted_articles = sort_articles_by_cves(articles)

    logger.info("Getting latest CVE")
    try:
        latest_cve = config_options.es_cve_client.query_documents(
            CVESearchQuery(limit=1, sort_by="modified_date", sort_order="desc"), True
        )[0][0]
    except IndexError:
        logger.error('No existing CVEs found, please run "create_cves" first')
        return

    _create_cves(latest_cve.modified_date, sorted_articles)

    logger.info("Querying CVEs for updating")
    cves_requiring_update = config_options.es_cve_client.query_documents(
        CVESearchQuery(limit=0, cves=set(sorted_articles.keys())), True
    )[0]

    logger.info(f"{len(cves_requiring_update)} CVEs to process")

    for cve in cves_requiring_update:
        relevant_articles = sorted_articles[cve.cve]
        cve.document_count = len(relevant_articles)
        cve.documents = {article.id for article in relevant_articles}
        cve.dating = {article.publish_date for article in relevant_articles}
        cve.keywords = get_article_common_keywords(relevant_articles)

    logger.info("Updating article details of CVEs")
    config_options.es_cve_client.update_documents(
        cves_requiring_update, ["document_count", "documents", "dating", "keywords"]
    )


@app.command()
def major_update_cves(start: str = "") -> None:
    start_date = datetime.fromisoformat(start) if len(start) > 0 else None

    logger.info("Downloading articles")
    articles = config_options.es_article_client.query_documents(
        ArticleSearchQuery(limit=0), False
    )[0]
    sorted_articles = sort_articles_by_cves(articles)

    total_new = 0

    for raw_cves in query_nvd(start_date):
        parsed_cves: list[FullCVE] = []

        for raw_cve in raw_cves:
            articles = []

            if raw_cve["cve"]["id"] in sorted_articles:
                articles = sorted_articles[raw_cve["cve"]["id"]]

            parsed_cves.append(validate_cve(raw_cve["cve"], articles))

        cve_ids = [cve.cve for cve in parsed_cves]
        existing_cves = {
            cve.cve: cve
            for cve in config_options.es_cve_client.query_documents(
                CVESearchQuery(limit=10000, cves=set(cve_ids)), True
            )[0]
        }

        new_cves: list[FullCVE] = []

        for parsed_cve in parsed_cves:
            if (
                parsed_cve.cve not in existing_cves
                or parsed_cve.publish_date != existing_cves[parsed_cve.cve].publish_date
            ):
                new_cves.append(parsed_cve)

        total_new += len(new_cves)

        logger.info(
            f"Found {len(new_cves)} new cves, with {total_new} total new so far"
        )

        if len(new_cves) == 0:
            logger.info("Skipping")
            continue

        logger.info(f"Creating CVE titles for {len(new_cves)} cves")
        new_cves = process_threaded(new_cves, generate_cve_title, 32)

        logger.info(f"Saving {len(new_cves)} CVEs")
        config_options.es_cve_client.save_documents(new_cves, chunk_size=500)


@app.command()
def title_cves() -> None:
    for i, cves in enumerate(
        config_options.es_cve_client.scroll_documents(
            CVESearchQuery(limit=0, sort_by="publish_date", sort_order="desc"),
            pit_keep_alive="30m",
            batch_size=1000,
        )
    ):
        logger.info(f"Processing cve batch {i} of {len(cves)} cves")

        untitled_cves = [cve for cve in cves if not cve.title]

        if len(untitled_cves) == 0:
            continue

        logger.info(f"Generating titles for {len(untitled_cves)} cves")
        titled_cves = process_threaded(untitled_cves, generate_cve_title, 32)

        logger.info("Updating CVEs")
        config_options.es_cve_client.update_documents(titled_cves, ["title"])


if __name__ == "__main__":
    app()

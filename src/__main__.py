import logging
from modules.objects import FullCluster, FullArticle

from src.cluster import create_clusters, cluster_new_articles

from .map import calc_cords, calc_similar

from . import config_options
from modules.elastic import ArticleSearchQuery

logger = logging.getLogger("osinter")


def map_articles(articles: list[FullArticle]):
    if articles is None:
        logger.info("Downloading articles for ML")

        articles = config_options.es_article_client.query_documents(
            ArticleSearchQuery(limit=0), True
        )[0]

    logger.info("Calculating cords")
    calc_cords(articles)

    logger.info("Grouping article by similarity")
    calc_similar(articles, 20)


def cluster_articles(articles: list[FullArticle], clusters: list[FullCluster]):
    logger.info("Clustering articles")
    cluster_new_articles(
        articles,
        clusters,
        saved_model="./models/topic_model",
    )


def update_articles():
    logger.info("Downloading articles")
    articles = config_options.es_article_client.query_documents(
        ArticleSearchQuery(limit=0), True
    )[0]

    logger.info("Downloading clusters")
    clusters = config_options.es_cluster_client.query_all_documents()

    map_articles(articles)
    cluster_articles(articles, clusters)

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


def create_topic_model():
    logger.info("Downloading articles")
    articles = config_options.es_article_client.query_documents(
        ArticleSearchQuery(limit=0), True
    )[0]

    logger.info("Generating clusters. This could take some time")
    new_clusters = create_clusters(
        articles, "./models/topic_model", bool(config_options.OPENAI_KEY)
    )

    # TODO: Remove old clusters
    logger.info(f"Generated {len(new_clusters)} clusters. Saving them")
    saved_clusters = config_options.es_cluster_client.save_documents(new_clusters, True)
    logger.info(f"Save {saved_clusters} clusters")

    map_articles(articles)

    logger.info(f"Updating {len(articles)} articles")

    updated_articles_count = config_options.es_article_client.update_documents(
        articles, ["ml", "similar"]
    )

    logger.info(f"Updated {updated_articles_count} articles")


create_topic_model()

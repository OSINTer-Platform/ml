from typing import cast
import umap
import numpy as np
from nptyping import Float64, Int64, NDArray, Float32, Shape

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import distance

from hdbscan import HDBSCAN

from modules.elastic import ArticleSearchQuery
from modules.objects import FullArticle
from . import config_options


def generate_embeddings(text_list: list[str]) -> NDArray[Shape["*, 2"], Float32]:
    tfidf_vectorizer = TfidfVectorizer(min_df=5, stop_words="english")
    tfidf_word_doc_matrix = tfidf_vectorizer.fit_transform(text_list)
    tfidf_embedding = umap.UMAP(min_dist=0, n_neighbors=10, metric="hellinger").fit(
        tfidf_word_doc_matrix
    )

    return cast(NDArray[Shape["*, 2"], Float32], tfidf_embedding.embedding_)


def cluster_articles(
    articles: list[FullArticle], numberOfNearest: int
) -> list[FullArticle]:
    text_list: list[str] = [article.content for article in articles]

    embeddings = generate_embeddings(text_list)

    distances: NDArray[Shape["*, 2"], Float64] = distance.squareform(
        distance.pdist(embeddings)
    )
    closest = np.argsort(distances, axis=1)[:, 1 : numberOfNearest + 1]

    clusters = cast(
        NDArray[Shape["*"], Int64],
        HDBSCAN(min_cluster_size=10, min_samples=5).fit(embeddings).labels_,
    )

    for i, article in enumerate(articles):
        article.ml = {
            "similar": [articles[point].id for point in closest[i]],
            "cluster": int(clusters[i]),
            "coordinates": (float(embeddings[i][0]), float(embeddings[i][1])),
        }

    return articles


def main() -> None:
    articles = config_options.es_article_client.query_documents(
        ArticleSearchQuery(limit=0), True
    )

    cluster_articles(articles, 20)

    config_options.es_article_client.save_documents(articles)


if __name__ == "__main__":
    main()

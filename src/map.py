from typing import cast
import umap
from nptyping import NDArray, Float32, Shape

from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

from modules.objects import FullArticle


def _dim_reduction(text_list: list[str]) -> NDArray[Shape["*, 2"], Float32]:
    tfidf_vectorizer = TfidfVectorizer(min_df=5, stop_words="english")
    tfidf_word_doc_matrix = tfidf_vectorizer.fit_transform(text_list)
    tfidf_embedding = umap.UMAP(min_dist=0, n_neighbors=10, metric="hellinger").fit(
        tfidf_word_doc_matrix
    )

    return cast(NDArray[Shape["*, 2"], Float32], tfidf_embedding.embedding_)


def calc_cords(articles: list[FullArticle]):
    contents = [article.content for article in articles]
    embeddings = _dim_reduction(contents)

    for i, article in enumerate(articles):
        article.ml["coordinates"] = (float(embeddings[i][0]), float(embeddings[i][1]))


def calc_similar(articles: list[FullArticle], numberOfNearest: int):
    """Relies on proper coordinates for all articles, so should be called AFTER calc_cords"""
    cords = [
        [article.ml["coordinates"][0], article.ml["coordinates"][1]]
        for article in articles
    ]
    _, closest = (
        NearestNeighbors(n_neighbors=numberOfNearest + 1, algorithm="brute")
        .fit(cords)
        .kneighbors(cords)
    )
    closest = closest[:, 1:]

    for i, article in enumerate(articles):
        article.similar = [articles[point].id for point in closest[i]]

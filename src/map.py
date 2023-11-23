from typing import Any, cast
from umap import UMAP
from nptyping import NDArray, Float32, Shape

from sklearn.neighbors import NearestNeighbors

from modules.objects import FullArticle


def _dim_reduction(
    embeddings: NDArray[Any, Any], model: UMAP | None
) -> tuple[NDArray[Shape["*, 2"], Float32], UMAP]:
    if not model:
        model = cast(
            UMAP,
            UMAP(min_dist=0, n_neighbors=7, n_components=2, metric="cosine", random_state=42).fit(
                embeddings
            ),
        )

    reduced_embeddings = model.transform(embeddings)
    return cast(NDArray[Shape["*, 2"], Float32], reduced_embeddings), model


def calc_cords(
    articles: list[FullArticle],
    embeddings: NDArray[Any, Any],
    model: UMAP | None = None,
) -> UMAP:
    reduced_embeddings, model = _dim_reduction(embeddings, model)

    for i, article in enumerate(articles):
        article.ml.coordinates = (
            float(reduced_embeddings[i][0]),
            float(reduced_embeddings[i][1]),
        )

    return model


def calc_similar(articles: list[FullArticle], numberOfNearest: int) -> None:
    """Relies on proper coordinates for all articles, so should be called AFTER calc_cords"""
    cords = [
        [article.ml.coordinates[0], article.ml.coordinates[1]]
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

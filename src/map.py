import logging
import pickle
from typing import Any, cast
from umap import UMAP
from nptyping import NDArray, Float32, Shape

from sklearn.neighbors import NearestNeighbors

from modules.objects import FullArticle

logger = logging.getLogger("osinter")

UMAP_MODEL_PATH = "./models/map_umap"


def calc_cords(
    articles: list[FullArticle], embeddings: NDArray[Any, Any], regenerate: bool
):
    if regenerate:
        logger.debug("Generating new umap model")
        umap = UMAP(
            min_dist=0, n_neighbors=7, n_components=2, metric="cosine", random_state=42
        )
        umap.fit(embeddings)

        logger.debug(f'Saving new umap model to "{UMAP_MODEL_PATH}"')
        with open(UMAP_MODEL_PATH, "wb") as f:
            pickle.dump(umap, f)

    else:
        with open(UMAP_MODEL_PATH, "rb") as f:
            umap: UMAP = pickle.load(f)

    reduced_embeddings = cast(
        NDArray[Shape["*, 2"], Float32], umap.transform(embeddings)
    )

    for i, article in enumerate(articles):
        article.ml.coordinates = (
            float(reduced_embeddings[i][0]),
            float(reduced_embeddings[i][1]),
        )


def calc_similar(articles: list[FullArticle], numberOfNearest: int) -> None:
    """Relies on proper coordinates for all articles, so should be called AFTER calc_cords"""
    cords = [
        [article.ml.coordinates[0], article.ml.coordinates[1]] for article in articles
    ]
    _, closest = (
        NearestNeighbors(n_neighbors=numberOfNearest + 1, algorithm="brute")
        .fit(cords)
        .kneighbors(cords)
    )
    closest = closest[:, 1:]

    for i, article in enumerate(articles):
        article.similar = [articles[point].id for point in closest[i]]

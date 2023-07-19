import umap
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.spatial import distance

from hdbscan import HDBSCAN

from modules import config, elastic, objects

config_options = config.BackendConfig()

def generate_embeddings(data):
    tfidf_vectorizer = TfidfVectorizer(min_df=5, stop_words='english')
    tfidf_word_doc_matrix = tfidf_vectorizer.fit_transform(data)
    tfidf_embedding = umap.UMAP(min_dist=0, n_neighbors=10, metric='hellinger').fit(tfidf_word_doc_matrix)

    return tfidf_embedding.embedding_

def main(articles, numberOfNearest):
    data = [article.content for article in articles]

    embeddings = generate_embeddings(data)
    distances = distance.squareform(distance.pdist(embeddings))
    closest = np.argsort(distances, axis=1)[:, 1:numberOfNearest+1]

    clusters = HDBSCAN(min_cluster_size=10, min_samples=5).fit(embeddings).labels_

    for i,article in enumerate(articles):
        article.ml = { "similar" : [articles[point].id for point in closest[i]], "cluster" : clusters[i] }

        config_options.es_article_client.save_document(article)

if __name__ == "__main__":
    articles = config_options.es_article_client.query_documents(elastic.SearchQuery(limit = 0, complete=True))["documents"]

    main(articles, 12)

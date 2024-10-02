# combined_search.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def combined_search(query_tfidf, query_embedding, tfidf_weight, sbert_weight, top_k=5, tfidf_matrix=None, index=None, filenames=None):
    if tfidf_matrix is None or index is None or filenames is None:
        print("Necessary data is not provided.")
        return []

    # TF-IDF benzerlik skorları
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    # SBERT benzerlik skorları
    distances, indices = index.search(query_embedding, len(filenames))
    sbert_similarities = np.zeros(len(filenames))
    for idx, distance in zip(indices[0], distances[0]):
        sbert_similarities[idx] = distance
    # Kombine skorları hesapla
    combined_scores = tfidf_weight * cosine_similarities + sbert_weight * sbert_similarities
    # En yüksek skorlu dokümanları bul
    top_indices = combined_scores.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        filename = filenames[idx]
        score = combined_scores[idx]
        results.append((filename, score))
    return results

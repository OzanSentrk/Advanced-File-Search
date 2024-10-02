# tfidf_search.py

from sklearn.metrics.pairwise import cosine_similarity

def tfidf_search(query_tfidf, top_k=5, tfidf_matrix=None, filenames=None):
    if tfidf_matrix is None or filenames is None:
        print("TF-IDF matrix or filenames list is not provided.")
        return []

    # Benzerlik skorlarını hesapla
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    # En yüksek skorlu dokümanları bul
    top_indices = cosine_similarities.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        filename = filenames[idx]
        score = cosine_similarities[idx]
        results.append((filename, score))
    return results

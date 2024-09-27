# tfidf_search.py

from sklearn.metrics.pairwise import cosine_similarity
import data_management  # Modülü import edin

def tfidf_search(query_tfidf, top_k=5):
    # Compute TF-IDF similarities
    tfidf_similarities = cosine_similarity(query_tfidf, data_management.tfidf_matrix).flatten()
    combined_scores = list(zip(data_management.filenames, tfidf_similarities))
    # Sort results
    combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)[:top_k]
    print("Top results using TF-IDF similarity:")
    for filename, score in combined_scores:
        print(f"File: {filename}, Score: {score:.4f}")

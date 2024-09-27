# combined_search.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import data_management  # Modülü import edin

def combined_search(query_tfidf, query_embedding, tfidf_weight, sbert_weight, top_k=5):
    # Compute TF-IDF similarities
    tfidf_similarities = cosine_similarity(query_tfidf, data_management.tfidf_matrix).flatten()
    # Compute combined scores
    combined_scores = []
    for idx in range(len(data_management.filenames)):
        sbert_file = data_management.filenames[idx]
        doc_embedding = data_management.index.reconstruct(idx)
        sbert_similarity = cosine_similarity(query_embedding, [doc_embedding]).flatten()[0]
        tfidf_similarity = tfidf_similarities[idx]
        combined_score = tfidf_weight * tfidf_similarity + sbert_weight * sbert_similarity
        combined_scores.append((sbert_file, combined_score))
    # Sort results
    combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)[:top_k]
    print(f"Top results using combined similarity (TF-IDF {int(tfidf_weight*100)}% + SBERT {int(sbert_weight*100)}%):")
    for filename, combined_score in combined_scores:
        print(f"File: {filename}, Score: {combined_score:.4f}")

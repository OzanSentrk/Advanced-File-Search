# tfidf_search.py

import numpy as np
import faiss

def tfidf_search(query_tfidf, index, top_k=5, filenames=None):
    if index is None or filenames is None:
        print("FAISS index veya dosya isimleri listesi sağlanmadı.")
        return []
    
    # Sorgu vektörünü float32 tipine dönüştürme
    query_vector = query_tfidf.astype('float32').toarray()
    
    # Sorgu vektörünü normalleştirme
    faiss.normalize_L2(query_vector)
    
    # FAISS ile arama yapma
    scores, indices = index.search(query_vector, top_k)
    
    results = []
    for idx, score in zip(indices[0], scores[0]):
        filename = filenames[idx]
        results.append((filename, score))
    return results

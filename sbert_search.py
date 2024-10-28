# sbert_search.py

def sbert_search(query_embedding, top_k=5, index=None, filenames=None):
    if index is None or filenames is None:
        
        return []

    # FAISS indeksinde benzer vektörleri bul
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        filename = filenames[idx]
        score = float(distance)
        results.append((filename, score))
    return results

# sbert_search.py

import numpy as np
import data_management  # Modülü import edin

def sbert_search(query_embedding, top_k=5):
    # Search in FAISS index
    distances, indices = data_management.index.search(query_embedding, top_k)
    print("Top results using SBERT similarity:")
    for idx, distance in zip(indices[0], distances[0]):
        if idx < len(data_management.filenames):
            filename = data_management.filenames[idx]
            print(f"File: {filename}, Score: {distance:.4f}")

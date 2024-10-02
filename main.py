# main.py

import os
from sklearn.exceptions import NotFittedError
import numpy as np
import data_management  # Modülü import edin
from data_management import (
    load_index_and_filenames,
    check_and_process_new_files,
    get_file_contents,
    vectorize_and_index_content,
    save_index_and_filenames,
    model,
)
from preprocessing import preprocess_text
from model_selection import select_model, get_similarity_weights
from tfidf_search import tfidf_search
from sbert_search import sbert_search
from combined_search import combined_search
from constants import samples_folder


# Function to request access
def request_access():
    user_input = input(f"Type 'yes' to grant access to '{samples_folder}': ").strip().lower()
    return user_input == 'yes'

def search_query(query, top_k=5):
    print("Processing query...")
    # Preprocess the query
    query_processed = preprocess_text(query)
    print(f"Preprocessed query: {query_processed}")
    # Vectorize the query using TF-IDF
    try:
        query_tfidf = data_management.tfidf_vectorizer.transform([query_processed])
    except NotFittedError:
        print("TF-IDF vectorizer is not fitted. Please vectorize documents first.")
        return
    # Encode the query using SBERT
    query_embedding = data_management.model.encode(query_processed, normalize_embeddings=True).astype('float32')
    query_embedding = np.array([query_embedding])
    # Check if index is empty
    if data_management.index.ntotal == 0:
        print("FAISS index is empty. Please vectorize documents first.")
        return
    # Determine which model to use
    selected_model = select_model(query)
    print(f"Selected model: {selected_model}")
    if selected_model == 'tfidf':
        tfidf_search(query_tfidf, top_k)
    elif selected_model == 'sbert':
        sbert_search(query_embedding, top_k)
    else:
        tfidf_weight, sbert_weight = get_similarity_weights(query)
        combined_search(query_tfidf, query_embedding, tfidf_weight, sbert_weight, top_k)


# main.py

if __name__ == "__main__":
    if request_access():
        # Load index and filenames
        data_loaded = load_index_and_filenames()
        if not data_loaded:
            print("Data could not be loaded, scanning and vectorizing files...")
            # First time, process all files
            data_management.preprocessed_texts, data_management.filenames = get_file_contents(samples_folder)
            data_management.tfidf_matrix = vectorize_and_index_content(data_management.preprocessed_texts)
            save_index_and_filenames(data_management.filenames, data_management.preprocessed_texts)
        else:
            # Check for new files and process
            check_and_process_new_files()
        # Get user query and search
        query = input("What would you like to search for?: ").strip()
        search_query(query, top_k=5)
    else:
        print("Access not granted, program terminated.")


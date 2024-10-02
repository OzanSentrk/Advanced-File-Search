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
    search_query,
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
        results = search_query(query, top_k=5)
        # Sonuçları konsola yazdırın
        if results:
            print("Arama Sonuçları:")
            for filename, score in results:
                print(f"Dosya: {filename}, Skor: {score}")
        else:
            print("Sonuç bulunamadı.")
    else:
        print("Access not granted, program terminated.")

# data_management.py

import os
import pickle
import pdfplumber
import faiss
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from constants import db_directory, samples_folder, dimension,turkish_stopwords
from preprocessing import preprocess_text
from docx import Document
import pandas as pd
from pptx import Presentation
from scipy.sparse import csr_matrix

# Initialize variables
tfidf_vectorizer = TfidfVectorizer(stop_words=list(turkish_stopwords))
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
print("Model loaded successfully.")
index = None  # Will be initialized later
filenames = []
tfidf_matrix = None
preprocessed_texts = []

# Function to read PDF files
def read_pdf(file_path):
    print(f"Reading PDF file: {file_path}")
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except Exception as e:
        print(f"Could not read PDF file: {file_path}, error: {e}")
    return text

def read_word(file_path):
    print(f"Word dosyası okunuyor: {file_path}")
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Word dosyası okunamadı: {file_path}, hata: {e}")
    return text

def read_excel(file_path):
    print(f"Excel dosyası okunuyor: {file_path}")
    text = ""
    try:
        # Tüm sayfaları oku
        xls = pd.ExcelFile(file_path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            # DataFrame'deki tüm hücreleri birleştir
            for column in df.columns:
                column_data = ' '.join(df[column].astype(str).tolist())
                text += column_data + "\n"
                print(f"Sheet: {sheet_name}, Column: {column}, Data: {column_data}")
    except Exception as e:
        print(f"Excel dosyası okunamadı: {file_path}, hata: {e}")
    return text


def read_powerpoint(file_path):
    print(f"PowerPoint dosyası okunuyor: {file_path}")
    text = ""
    try:
        prs = Presentation(file_path)
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
    except Exception as e:
        print(f"PowerPoint dosyası okunamadı: {file_path}, hata: {e}")
    return text


# Function to get file contents
def get_file_contents(folder):
    print(f"'{folder}' klasörü taranıyor...")
    files_content = []
    filenames = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"Dosya işleniyor: {file_path}")
            content_raw = ""
            if file.endswith('.txt'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content_raw = f.read()
                        print(f"Metin dosyası okundu: {file_path}")
                except Exception as e:
                    print(f"Metin dosyası okunamadı: {file_path}, hata: {e}")
            elif file.endswith('.pdf'):
                content_raw = read_pdf(file_path)
            elif file.endswith(('.docx', '.doc')):
                content_raw = read_word(file_path)
            elif file.endswith(('.xlsx', '.xls')):
                content_raw = read_excel(file_path)
            elif file.endswith(('.pptx', '.ppt')):
                content_raw = read_powerpoint(file_path)
            else:
                print(f"Desteklenmeyen dosya türü: {file_path}")
                continue  # Desteklenmeyen dosyaları atla
            # Metin ön işlemesi
            content = preprocess_text(content_raw)
            if content:
                files_content.append(content)
                filenames.append(file)
    print(f"Taranan dosya sayısı: {len(filenames)}")
    return files_content, filenames

# Function to vectorize content and build index
def vectorize_and_index_content(contents):
    global index, tfidf_vectorizer, tfidf_matrix
    # Initialize FAISS index
    index = faiss.IndexFlatIP(dimension)
    print("Vectorizing files and adding to FAISS index (SBERT and TF-IDF)...")
    # TF-IDF vectorization
    tfidf_matrix = tfidf_vectorizer.fit_transform(contents)
    # SBERT embeddings
    with tqdm(total=len(contents)) as pbar:
        for content in contents:
            embedding = model.encode(content, normalize_embeddings=True).astype('float32')
            index.add(np.array([embedding]))
            pbar.update(1)
    print("Vectorization completed and added to FAISS index.")
    return tfidf_matrix

# Function to save index and filenames
def save_index_and_filenames(filenames, preprocessed_texts):
    faiss.write_index(index, os.path.join(db_directory, 'faiss_index.index'))
    with open(os.path.join(db_directory, 'filenames.pkl'), 'wb') as f:
        pickle.dump(filenames, f)
    with open(os.path.join(db_directory, 'tfidf_matrix.pkl'), 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    with open(os.path.join(db_directory, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(os.path.join(db_directory, 'preprocessed_texts.pkl'), 'wb') as f:
        pickle.dump(preprocessed_texts, f)
    print("FAISS index, TF-IDF matrix, vectorizer, preprocessed texts, and filenames saved.")

# Function to load index and filenames
def load_index_and_filenames():
    global index, tfidf_vectorizer, tfidf_matrix, filenames, preprocessed_texts
    faiss_index_path = os.path.join(db_directory, 'faiss_index.index')
    filenames_path = os.path.join(db_directory, 'filenames.pkl')
    tfidf_matrix_path = os.path.join(db_directory, 'tfidf_matrix.pkl')
    tfidf_vectorizer_path = os.path.join(db_directory, 'tfidf_vectorizer.pkl')
    preprocessed_texts_path = os.path.join(db_directory, 'preprocessed_texts.pkl')
    if all(os.path.exists(path) for path in [faiss_index_path, filenames_path, tfidf_matrix_path, tfidf_vectorizer_path, preprocessed_texts_path]):
        index = faiss.read_index(faiss_index_path)
        with open(filenames_path, 'rb') as f:
            filenames = pickle.load(f)
        with open(tfidf_matrix_path, 'rb') as f:
            tfidf_matrix = pickle.load(f)
        with open(tfidf_vectorizer_path, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        with open(preprocessed_texts_path, 'rb') as f:
            preprocessed_texts = pickle.load(f)
        print("FAISS index, TF-IDF vectorizer, TF-IDF matrix, preprocessed texts, and filenames loaded.")
        return True
    else:
        index = faiss.IndexFlatIP(dimension)
        return False

# Function to check and process new files
def check_and_process_new_files():
    global filenames, preprocessed_texts, tfidf_matrix
    remove_deleted_files()
    # Get current files in the samples_folder
    current_files = set()
    for root, dirs, files in os.walk(samples_folder):
        for file in files:
            current_files.add(file)
    # Find new files
    new_files = current_files - set(filenames)
    if new_files:
        new_files_content = []
        new_files_names = []
        for file in new_files:
            file_path = os.path.join(samples_folder, file)
            print(f"Yeni dosya işleniyor: {file_path}")
            content_raw = ""
            if file.endswith('.txt'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content_raw = f.read()
                        print(f"Metin dosyası okundu: {file_path}")
                except Exception as e:
                    print(f"Metin dosyası okunamadı: {file_path}, hata: {e}")
            elif file.endswith('.pdf'):
                content_raw = read_pdf(file_path)
            elif file.endswith(('.docx', '.doc')):
                content_raw = read_word(file_path)
            elif file.endswith(('.xlsx', '.xls')):
                content_raw = read_excel(file_path)
            elif file.endswith(('.pptx', '.ppt')):
                content_raw = read_powerpoint(file_path)
            else:
                print(f"Desteklenmeyen dosya türü: {file_path}")
                continue  # Desteklenmeyen dosyaları atla
            # Metin ön işlemesi
            content = preprocess_text(content_raw)
            if content:
                new_files_content.append(content)
                new_files_names.append(file)
        # Combine with existing data
        all_contents = preprocessed_texts + new_files_content
        all_filenames = filenames + list(new_files_names)
        # Re-vectorize and rebuild index
        tfidf_matrix = vectorize_and_index_content(all_contents)
        # Update global variables
        filenames = all_filenames
        preprocessed_texts = all_contents
        # Save updated data
        save_index_and_filenames(filenames, preprocessed_texts)
    else:
        print("Yeni dosya bulunamadı.")

def remove_deleted_files():
    global filenames, preprocessed_texts, tfidf_matrix
    # Get current files in the samples_folder
    current_files = set()
    for root, dirs, files in os.walk(samples_folder):
        for file in files:
            current_files.add(file)
    # Find deleted files
    deleted_files = set(filenames) - current_files
    if deleted_files:
        print(f"Silinen dosyalar tespit edildi: {deleted_files}")
        # Indeks ve veri yapılarından silinen dosyaları kaldır
        indices_to_remove = [idx for idx, filename in enumerate(filenames) if filename in deleted_files]
        # Indeksleri sıralamaya dikkat edin
        indices_to_remove.sort()
        print(f"Silinecek indeksler: {indices_to_remove}")
        # FAISS indeksinden vektörleri kaldır
        remove_vectors_from_index(indices_to_remove)
        # filenames, preprocessed_texts ve tfidf_matrix'ten ilgili verileri kaldır
        filenames = [filename for idx, filename in enumerate(filenames) if idx not in indices_to_remove]
        preprocessed_texts = [text for idx, text in enumerate(preprocessed_texts) if idx not in indices_to_remove]
        # TF-IDF matrisini güncelle
        tfidf_matrix = delete_rows_csr(tfidf_matrix, indices_to_remove)
        # Güncellenmiş verileri kaydet
        save_index_and_filenames(filenames, preprocessed_texts)
        print("Silinen dosyalar indeksten ve veri yapılarından kaldırıldı.")
    else:
        print("Silinen dosya bulunamadı.")
def remove_vectors_from_index(indices_to_remove):
    global index
    # FAISS indeksinde kalan vektörleri al
    total_vectors = index.ntotal
    if total_vectors == 0:
        return
    remaining_indices = [i for i in range(total_vectors) if i not in indices_to_remove]
    if remaining_indices:
        # Kalan vektörleri al
        remaining_vectors = index.reconstruct_n(0, total_vectors)
        remaining_vectors = remaining_vectors[remaining_indices]
        # Yeni bir indeks oluştur ve kalan vektörleri ekle
        index = faiss.IndexFlatIP(dimension)
        index.add(remaining_vectors)
    else:
        # Tüm vektörler silindiyse yeni boş bir indeks oluştur
        index = faiss.IndexFlatIP(dimension)
def delete_rows_csr(mat, indices):
    if not isinstance(mat, csr_matrix):
        raise ValueError("Only CSR format is supported")
    indices = list(sorted(set(indices)))
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]
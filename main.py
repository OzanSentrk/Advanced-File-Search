# main.py

import os
from data_management import (
    load_index_and_filenames,
    check_and_process_new_files,
    get_file_contents,
    vectorize_and_index_content,
    save_index_and_filenames,
    search_query,
    load_sbert_model
)
from constants import samples_folder

# Function to request access
def request_access():
    user_input = input(f"'{samples_folder}' klasörüne erişim izni vermek için 'evet' yazın: ").strip().lower()
    return user_input == 'evet'

if __name__ == "__main__":
    if request_access():
        # SBERT modelini yükle
        load_sbert_model()
        
        # İndeks ve dosya isimlerini yükle
        data_loaded = load_index_and_filenames()
        if not data_loaded:
            print("Veriler yüklenemedi, dosyalar taranıyor ve vektörleştiriliyor...")
            # İlk kez çalıştırılıyorsa, tüm dosyaları işle
            contents, filenames = get_file_contents(samples_folder)
            vectorize_and_index_content(contents)
            save_index_and_filenames(filenames, contents)
        else:
            # Yeni dosyaları kontrol et ve işle
            check_and_process_new_files()
        # Kullanıcıdan sorgu al ve ara
        query = input("Ne aramak istersiniz?: ").strip()
        results, elapsed_time = search_query(query, top_k=5)
        # Sonuçları konsola yazdır
        if results:
            print("Arama Sonuçları:")
            for filename, score in results:
                print(f"Dosya: {filename}, Skor: {score:.4f}")
            print(f"Arama süresi: {elapsed_time:.4f} saniye")
        else:
            print("Sonuç bulunamadı.")
    else:
        print("Erişim izni verilmedi, program sonlandırıldı.")

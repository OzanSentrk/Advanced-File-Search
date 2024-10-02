# gui.py

import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QTextEdit, QVBoxLayout
from PyQt6.QtCore import Qt
from data_management import search_query
from data_management import (
    load_index_and_filenames,
    check_and_process_new_files,
    get_file_contents,
    vectorize_and_index_content,
    save_index_and_filenames,
    samples_folder
)

class SearchApp(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dosya Arama Uygulaması")
        self.setGeometry(100, 100, 800, 600)
        self.initUI()

    def initUI(self):
        # Ana dikey düzen
        self.layout = QVBoxLayout()

        # Arama etiketini ekleyelim
        self.search_label = QLabel('Aranacak Kelime:')
        self.layout.addWidget(self.search_label)

        # Arama metin girişini ekleyelim
        self.search_input = QLineEdit()
        self.layout.addWidget(self.search_input)

        # Arama butonunu ekleyelim
        self.search_button = QPushButton('Ara')
        self.search_button.clicked.connect(self.search_files)
        self.layout.addWidget(self.search_button)

        # Sonuçları göstermek için metin alanı
        self.result_area = QTextEdit()
        self.result_area.setReadOnly(True)
        self.layout.addWidget(self.result_area)

        # Layout'u ayarla
        self.setLayout(self.layout)

    def search_files(self):
        query = self.search_input.text()
        if not query:
            self.result_area.setText("Lütfen arama terimini girin.")
            return

        # Arama işlemi sırasında arayüzün donmaması için
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            results = self.perform_search(query)
            if results:
                self.display_results(results)
            else:
                self.result_area.setText("Sonuç bulunamadı.")
        finally:
            QApplication.restoreOverrideCursor()

    def perform_search(self, query):
        # search_query fonksiyonunu çağırıyoruz
        top_k = 10  # Gösterilecek sonuç sayısı
        results = search_query(query, top_k=top_k)
        return results

    def display_results(self, results):
        self.result_area.clear()
        for filename, score in results:
            self.result_area.append(f"Dosya: {filename}, Skor: {score:.4f}")

# Burada initialize_data fonksiyonunu tanımlıyoruz
def initialize_data():
    # Verileri başlatma işlemleri
    if load_index_and_filenames():
        print("Veriler yüklendi.")
        # Yeni dosyaları kontrol edin
        check_and_process_new_files()
    else:
        print("Veriler yüklenemedi, dosyalar taranıyor ve vektörleştiriliyor...")
        # Dosyaları işle ve vektörleştir
        preprocessed_texts, filenames = get_file_contents(samples_folder)
        tfidf_matrix = vectorize_and_index_content(preprocessed_texts)
        save_index_and_filenames(filenames, preprocessed_texts)

if __name__ == '__main__':
    initialize_data()

    app = QApplication(sys.argv)
    window = SearchApp()
    window.show()
    sys.exit(app.exec())

# gui.py

import sys
import os
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout,
    QMessageBox, QListWidget, QHBoxLayout, QLabel, QFileDialog, QLineEdit, QTextEdit, QProgressBar,QListWidgetItem,QSpacerItem
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings
from data_management import (
    INDEX_BASE_DIR,
    load_index_and_filenames,
    rescan_folder,
    get_filenames,
    add_manual_documents,
    search_query,
    reset_globals,
    get_scanned_folder
    
)
from PyQt5.QtGui import QCursor
from PyQt5 import QtGui, QtCore
# WorkerThread sınıfı (İndeksleme işlemleri için)
class WorkerThread(QThread):
    finished = pyqtSignal(bool)  # İşlem tamamlandığında sinyal ve indeksleme yapılıp yapılmadığını bildirir
    error = pyqtSignal(str)      # Hata durumunda sinyal

    def __init__(self, folder):
        super().__init__()
        self.folder = folder

    def run(self):
        try:
            indexing_performed = load_index_and_filenames(self.folder)
            self.finished.emit(indexing_performed)
        except Exception as e:
            self.error.emit(str(e))

# RescanThread sınıfı (Yeniden tarama işlemleri için)
class RescanThread(QThread):
    finished = pyqtSignal()     # İşlem tamamlandığında sinyal
    error = pyqtSignal(str)     # Hata durumunda sinyal

    def __init__(self, folder):
        super().__init__()
        self.folder = folder

    def run(self):
        try:
            rescan_folder(self.folder)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

# LoadingWindow sınıfı
class LoadingWindow(QMainWindow):
    loading_finished_signal = pyqtSignal(bool)  # Loading tamamlandığında MainApp'a sinyal gönderir (indeksleme yapıldı mı?)
    loading_error_signal = pyqtSignal(str)      # Hata durumunda MainApp'a sinyal gönderir

    def __init__(self, folder):
        super().__init__()
        self.folder = folder
        self.init_ui()

    def init_ui(self):
        # Ana widget oluştur
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Ana layout oluştur
        layout = QVBoxLayout()

        # Yükleme mesajı
        self.loading_label = QLabel("Belgeler taranıyor ve indeksleniyor...")
        self.loading_label.setAlignment(Qt.AlignCenter)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(0)  # Belirsiz progress bar

        # Layout'a widget'ları ekleyin
        layout.addWidget(self.loading_label)
        layout.addWidget(self.progress_bar)

        # Ana widget'a layout'u ayarlayın
        central_widget.setLayout(layout)

        # Pencere başlığını ve boyutunu ayarlayın
        self.setWindowTitle("Yükleniyor")
        self.resize(400, 200)

        # Worker thread başlat
        self.worker_thread = WorkerThread(self.folder)
        self.worker_thread.finished.connect(self.loading_finished)
        self.worker_thread.error.connect(self.loading_error)
        self.worker_thread.start()

    def loading_finished(self, indexing_performed):
        if indexing_performed:
            QMessageBox.information(self, "Bilgi", "Belgeler tarandı ve indekslendi.")
        else:
            QMessageBox.information(self, "Bilgi", "Belgeler yüklendi.")
        self.loading_finished_signal.emit(indexing_performed)
        self.close()

    def loading_error(self, error_message):
        QMessageBox.critical(self, "Hata", f"İndeksleme sırasında bir hata oluştu: {error_message}")
        self.loading_error_signal.emit(error_message)
        self.close()

# DocumentsWindow sınıfı
class DocumentsWindow(QMainWindow):
    def __init__(self, parent):
        super().__init__(parent)
        self.selected_folder = parent.selected_folder
        self.init_ui()
        self.check_index()
        self.setStyleSheet('''
            QPushButton#anasayfa_btn {
            font-family:Arial;
            font-size:20px;
        }
            QPushButton#anasayfa_btn:hover {
            background-color: #b0afac;
            border: 3px solid white; 
            font-size:20px;
            text-align: center;
            vertical-align: middle;
            }
             QLabel#SeciliKlasorLabel2 {
                font-family:Arial;
                font-size:13px;
                font-weight:bold;
                                      }
        ''')

    def init_ui(self):
        # Ana widget oluştur
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Ana layout oluştur
        self.main_layout = QVBoxLayout()

        # Ana Sayfa butonu
        self.home_button = QPushButton("Ana Sayfa")
        self.home_button.setObjectName("anasayfa_btn")
        self.home_button.clicked.connect(self.go_home)
        self.home_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))

        # Seçili klasörü gösteren label
        self.selected_folder_label = QLabel(f"Seçili Klasör: {self.selected_folder}")
        self.selected_folder_label.setAlignment(Qt.AlignCenter)
        self.selected_folder_label.setObjectName("SeciliKlasorLabel2")

        # Belgeleri listelemek için QListWidget
        self.documents_list = QListWidget()
        self.load_documents()

        # Belge ekle butonu
        self.add_document_button = QPushButton("Belge Ekle")
        self.add_document_button.setObjectName("dokuman_ekle_btn")
        self.add_document_button.clicked.connect(self.add_document)
        self.add_document_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))

        # Yeniden Tara butonu
        self.rescan_button = QPushButton("Yeniden Tara")
        self.rescan_button.setObjectName("yeniden_tara_btn")
        self.rescan_button.clicked.connect(self.rescan_folder)
        self.rescan_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))

        # Butonları içeren layout
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.addWidget(self.add_document_button)
        self.buttons_layout.addWidget(self.rescan_button)

        # Layout'a widget'ları ekleyin
        self.main_layout.addWidget(self.home_button)
        self.main_layout.addWidget(self.selected_folder_label)
        self.main_layout.addLayout(self.buttons_layout)
        self.main_layout.addWidget(self.documents_list)

        # Ana widget'a layout'u ayarlayın
        self.central_widget.setLayout(self.main_layout)

        # Pencere başlığını ve boyutunu ayarlayın
        self.setWindowTitle("Hedef Klasördeki Belgeler")
        self.resize(600, 400)

    def check_index(self):
        # İndekslerin mevcut olup olmadığını kontrol edin
        folder_name = os.path.basename(os.path.normpath(self.selected_folder))
        db_directory = INDEX_BASE_DIR / folder_name
        if not db_directory.exists():
            # İndeksler silinmişse
            self.selected_folder_label.setText("Bir Dosya Yolu Seçilmemiştir. Lütfen Bir Dosya Yolu Seçiniz.")
            self.add_document_button.setEnabled(False)
            self.rescan_button.setEnabled(False)
            self.load_documents()  # Boş listeyi göster
        else:
            # İndeksler mevcutsa
            self.selected_folder_label.setText(f"Seçili Klasör: {self.selected_folder}")
            self.add_document_button.setEnabled(True)
            self.rescan_button.setEnabled(True)
            self.load_documents()

    def load_documents(self):
        try:
            folder_name = os.path.basename(os.path.normpath(self.selected_folder)) 
            db_directory = INDEX_BASE_DIR / folder_name
            if not db_directory.exists():
                QMessageBox.warning(self, "Uyarı", "İndeks dizini bulunamadı. Lütfen yeniden tarama yapın.")
                self.documents_list.clear()
                return
            # Taranmış belgelerin isimlerini yükleyin
            self.documents_list.clear()
            current_filenames = get_filenames()
           
            if not current_filenames:
                self.documents_list.addItem("Bu klasörde belge bulunamadı veya indeksler silinmiş.")
            for filename in current_filenames:
                item = QListWidgetItem()
                widget = QWidget()
                layout = QHBoxLayout()
                # Dosya adı label'ı
                label = QLabel(filename)
                label.setStyleSheet("font-size:15px; font-family:Arial;  ")
                # Dosyayı Aç butonu
                open_button = QPushButton("Dosyayı Aç")
                open_button.setObjectName("open_button3")
                open_button.clicked.connect(lambda checked, f=filename: self.open_file(f))
                open_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
                open_button.setFixedHeight(30)
                open_button.setFixedWidth(120)
                open_button.setStyleSheet("""
                QPushButton#open_button3 {
                      
                    background-color: qlineargradient(spread:pad, x1:0.057, y1:0.505682, x2:0.988636, y2:0.489, 
                    stop:0 rgba(139, 198, 236, 1), 
                    stop:1 rgba(149, 153, 226, 1)); 
                    color: black;  
                    border-radius: 10px; 
                    font-size: 15px; 
                    text-align: center;
                    vertical-align: middle;
                    padding:3px 3px;
                                    }
                QPushButton#open_button3:hover{
                border:1px solid black;

                }
                """)
                # Layout ayarları
                layout.addWidget(label)
                layout.addStretch()
                layout.addWidget(open_button)
                layout.setContentsMargins(5, 0, 5, 0)
                widget.setLayout(layout)
                self.documents_list.addItem(item)
                self.documents_list.setItemWidget(item, widget)
                self.documents_list.setStyleSheet('''
                QListWidget::item {
                    height:30px;
                    padding: 8px;         
                }
                ''')
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Belgeler yüklenirken bir hata oluştu: {e}")
            

    def open_file(self, filename):
        file_path = os.path.join(self.selected_folder, filename)
        if os.path.exists(file_path):
            try:
                if sys.platform == "win32":
                    os.startfile(file_path)
                elif sys.platform == "darwin":
                    os.system("open " + file_path)
                else:
                    subprocess.Popen(["xdg-open", file_path])
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Dosya açılırken bir hata oluştu: {e}")
        else:
            QMessageBox.warning(self, "Uyarı", "Dosya mevcut değil veya taşınmış.")
    def add_document(self):
        # Belge ekleme işlevselliği
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_paths, _ = file_dialog.getOpenFileNames(self, "Belgeleri Seç")
        if file_paths:
            add_manual_documents(file_paths, self.selected_folder)
            self.load_documents()
            QMessageBox.information(self, "Bilgi", "Belgeler eklendi ve indeks güncellendi.")
        else:
            QMessageBox.warning(self, "Uyarı", "Belge seçilmedi.")

    def rescan_folder(self):
        # "Yeniden Tara" butonuna tıklanınca çalışacak fonksiyon
        self.rescan_button.setEnabled(False)  # Butonu devre dışı bırak
        self.rescan_button.setText("Taranıyor...")

        self.rescan_thread = RescanThread(self.selected_folder)
        self.rescan_thread.finished.connect(self.on_rescan_finished)
        self.rescan_thread.error.connect(self.on_rescan_error)
        self.rescan_thread.start()

    def on_rescan_finished(self):
        # Rescan işlemi tamamlandığında çağrılır
        self.load_documents()  # Dosya listesini yeniden yükle
        QMessageBox.information(self, "Bilgi", "Klasör başarıyla yeniden tarandı ve indeks güncellendi.")
        self.rescan_button.setEnabled(True)  # Butonu tekrar etkinleştir
        self.rescan_button.setText("Yeniden Tara")

    def on_rescan_error(self, error_message):
        QMessageBox.critical(self, "Hata", f"Yeniden tarama sırasında bir hata oluştu: {error_message}")
        self.rescan_button.setEnabled(True)  # Butonu tekrar etkinleştir
        self.rescan_button.setText("Yeniden Tara")

    def go_home(self):
        self.parent().show()
        self.close()

# SearchApp sınıfı
class SearchApp(QMainWindow):
    def __init__(self, parent):
        super().__init__(parent)
        self.selected_folder = parent.selected_folder
        self.init_ui()
        self.check_index()
        self.setStyleSheet('''
            
            QPushButton#search_button2 {
                font-family:Arial;
                font-size:13px;
                font-weight:Bold;
                     }      
            QPushButton#search_button2:hover {
                 border:1px solid white;} 
                
            QLineEdit#search_input {
                font-family:Arial;
                border-radius:10px;
                border: 2px solid #81b9c9;
                background-color: white;
                height:30px;
                font-size:15px;
                           }
            QLineEdit#topk_input {
                font-family:Arial;
                border-radius:10px;
                border: 2px solid #81b9c9;
                background-color: white;
                height:30px;
                font-size:15px;
                width:15px;
                }
            QLabel#SeciliKlasorLabel {
                font-family:Arial;
                font-size:13px;
                font-weight:bold;
                                      }
            
                           
            ''')
        
        

    def init_ui(self):
        # Ana widget oluştur
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Ana layout oluştur
        self.main_layout = QVBoxLayout()

        # Ana Sayfa butonu
        self.home_button = QPushButton("Ana Sayfa")
        self.home_button.clicked.connect(self.go_home)
        self.home_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))

        # Seçili klasörü gösteren label
        self.selected_folder_label = QLabel()
        self.selected_folder_label.setObjectName("SeciliKlasorLabel")
        self.selected_folder_label.setAlignment(Qt.AlignCenter)

        # Arama kutusu ve butonları
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Arama sorgunuzu girin")
        self.search_input.setObjectName("search_input")
        self.search_button = QPushButton("Ara")
        self.search_button.clicked.connect(self.search_files)
        self.search_button.setObjectName("search_button2")

        self.top_k_input = QLineEdit()
        self.top_k_input.setObjectName("topk_input")
        self.top_k_input.setPlaceholderText("Sonuç sayısı")
        self.top_k_input.setFixedWidth(150)

        # Arama kutusu ve butonları için bir layout oluşturun
        search_layout = QHBoxLayout()
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.top_k_input)
        search_layout.addWidget(self.search_button)

        # Sonuç alanını oluşturun
        self.result_list = QListWidget()
        self.result_list.setObjectName("sonuçlar_listesi")

        # Layout'lara widget'ları ekleyin
        self.main_layout.addWidget(self.home_button)
        self.main_layout.addWidget(self.selected_folder_label)
        self.main_layout.addLayout(search_layout)
        self.main_layout.addWidget(self.result_list)

        # Ana widget'a layout'u ayarlayın
        self.central_widget.setLayout(self.main_layout)

        # Pencere başlığını ve boyutunu ayarlayın
        self.setWindowTitle("Arama Yap")
        self.resize(800, 600)

    def check_index(self):
        folder_name = os.path.basename(os.path.normpath(self.selected_folder)) 
        db_directory = INDEX_BASE_DIR / folder_name
        if not db_directory.exists():
            self.selected_folder_label.setText(f"Seçili Klasör: {self.selected_folder} (İndeks bulunamadı).Lütfen yeniden tarama yapın.")
            self.search_button.setEnabled(False)
        else:
            self.selected_folder_label.setText(f"Seçili Klasör: {self.selected_folder}")
            self.search_button.setEnabled(True)
    def perform_search(self, query):
        folder_name = os.path.basename(os.path.normpath(self.selected_folder)) 
        db_directory = INDEX_BASE_DIR / folder_name
        if not db_directory.exists():
            QMessageBox.warning(self, "Uyarı", "İndeks dizini bulunamadı. Lütfen yeniden tarama yapın.")
            return [], 0.0
        top_k_text = self.top_k_input.text()
        if top_k_text.isdigit():
            top_k = int(top_k_text)
        else:
            top_k = 9

        results, elapsed_time = search_query(query, top_k=top_k)
        return results, elapsed_time

    def search_files(self):
        query = self.search_input.text()
        if not query:
            QMessageBox.warning(self, "Uyarı", "Lütfen bir arama sorgusu girin.")
            return
        self.statusBar().showMessage("Arama yapılıyor...")
        QApplication.processEvents()

        results, elapsed_time = self.perform_search(query)

        self.display_results(results)
        self.statusBar().showMessage("Arama tamamlandı.")

        # Arama süresini konsola yazdır
       

    def display_results(self, results):
        self.result_list.clear()

        if not results:
            item = QListWidgetItem("Sonuç bulunamadı.")
            self.result_list.addItem(item)
            return

        for filename, score in results:
            item = QListWidgetItem()
            widget = QWidget()
            layout = QHBoxLayout()
            # Dosya adı ve skor
            label = QLabel(f"Dosya: {filename}, Skor: {score:.4f}")
            
            label.setStyleSheet("font-size:15px; font-family:Arial;  ")
            # Dosyayı Aç butonu
            open_button = QPushButton("Dosyayı Aç")
            open_button.setObjectName("open_button2")
            open_button.clicked.connect(lambda checked, f=filename: self.open_file(f))
            open_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
            open_button.setFixedHeight(30)
            open_button.setFixedWidth(120)
            
            open_button.setStyleSheet("""
                QPushButton#open_button2 {
                      
                    background-color: qlineargradient(spread:pad, x1:0.057, y1:0.505682, x2:0.988636, y2:0.489, 
                    stop:0 rgba(139, 198, 236, 1), 
                    stop:1 rgba(149, 153, 226, 1)); 
                    color: black;  
                    border-radius: 10px; 
                    font-size: 15px; 
                    text-align: center;
                    vertical-align: middle;
                    padding:3px 3px;
                                    }
                QPushButton#open_button2:hover{
                border:1px solid black;

                }
                """)
            # Layout ayarları
            layout.addWidget(label)
            layout.addStretch()
            
            
            layout.addWidget(open_button)
            layout.setContentsMargins(5, 0, 5, 0)
            widget.setLayout(layout)
            self.result_list.addItem(item)  # Boşluk eklemek için
            self.result_list.setItemWidget(item, widget)
            self.result_list.setStyleSheet('''
                QListWidget::item {
                    height:30px;
                    padding: 8px;         
                }
                ''')
    def open_file(self, filename):
        file_path = os.path.join(self.selected_folder, filename)
        if os.path.exists(file_path):
            try:
                if sys.platform == "win32":
                    os.startfile(file_path)
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", file_path])
                else:
                    subprocess.Popen(["xdg-open", file_path])
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Dosya açılamadı: {e}")

        else:
            QMessageBox.warning(self, "Uyarı", "Dosya mevcut değil veya taşınmış.")
    def go_home(self):
        self.parent().show()
        self.close()
class ScannedFoldersWindow(QMainWindow):
    def __init__(self, parent):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        # Ana widget ve layout oluştur
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout()

        # Ana Sayfa butonu
        self.home_button = QPushButton("Ana Sayfa")
        self.home_button.clicked.connect(self.go_home)
        self.home_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))

        # Taranmış klasörleri listelemek için QListWidget
        self.folders_list = QListWidget()
        self.load_scanned_folders()

        # Sil butonu
        self.delete_button = QPushButton("Seçili Klasörü Sil")
        self.delete_button.clicked.connect(self.delete_selected_folder)
        self.delete_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))

        # Layout'a widget'ları ekleyin
        self.main_layout.addWidget(self.home_button)
        self.main_layout.addWidget(self.folders_list)
        self.main_layout.addWidget(self.delete_button)

        self.central_widget.setLayout(self.main_layout)

        # Pencere başlığı ve boyutu
        self.setWindowTitle("Taranmış Klasörler")
        self.resize(600, 400)

    def load_scanned_folders(self):
        # Taranmış klasörleri yükleyin
        self.folders_list.clear()
        self.scanned_folders = get_scanned_folder()  # Fonksiyonu kullanıyoruz
        for folder_info in self.scanned_folders:
            folder_path = folder_info['path']
            index_size = folder_info['size']
            if folder_path:
                item_text = f"Klasör: {folder_path}, Boyut: {index_size / (1024*1024):.2f} MB"
            else:
                item_text = f"Klasör: {folder_info['name']}, Bilgi mevcut değil"
            self.folders_list.addItem(item_text)
            self.folders_list.setStyleSheet("""
        QListWidget::item {
            padding: 5px;
            border: 1px solid #ccc;
            border-radius: 5px;
            font-size: 15px;
            font-family: Arial;
            }
        QListWidget::item:selected {
            background-color: #8BC6EC;
            color: white;
            font-size: 15px;
            font-family: Arial;
            }
                """)

    def delete_selected_folder(self):
        selected_items = self.folders_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Uyarı", "Lütfen silmek istediğiniz klasörü seçin.")
            return
        for item in selected_items:
            row = self.folders_list.row(item)
            folder_info = self.scanned_folders[row]
            # İndeks dosyalarını silin
            folder_name = folder_info['name']
            db_directory = INDEX_BASE_DIR / folder_name
            if db_directory.exists():
                for file in db_directory.glob('*'):
                    file.unlink()
                db_directory.rmdir()
                
            
            # Listeden kaldırın
            self.folders_list.takeItem(row)
            # İlgili klasörü scanned_folders listesinden kaldırın
            self.scanned_folders.pop(row)
            QMessageBox.information(self, "Bilgi", "Seçili klasör ve indeksleri silindi.")
            self.parent().on_index_deleted(folder_name)

    def go_home(self):
        self.parent().show()
        self.close()


          
# MainApp sınıfı
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.selected_folder = None  
        self.index_loaded = False
        self.init_ui()
        self.prompt_folder_selection()
        self.setStyleSheet('''
            QMainWindow {
                background-color: lightblue;
            }
            QPushButton {
                color:black;
                background-color: #f7edf2;
                padding:15px;
                margin-bottom:5px;
                border-radius:15px;
                font-family:Arial;
                font-size:15px;
                font-weight:bold;

            }
            
            QPushButton:hover {
                background-color: #b0afac;
                border: 3px solid white;   }
            
                  
            ''')

    def init_ui(self):
        # Ana widget oluştur
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Ana layout oluştur
        main_layout = QVBoxLayout()

        # Arama yap butonu
        self.search_button = QPushButton("Arama Yap")
        self.search_button.clicked.connect(self.open_search_window)
        self.search_button.setObjectName("search_button")
        self.search_button.setEnabled(False)  # Başlangıçta devre dışı
        self.search_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))

        # Hedef klasörü değiştirme butonu
        self.change_folder_button = QPushButton("Hedef Klasörü Seç")
        self.change_folder_button.clicked.connect(self.change_target_folder)
        self.change_folder_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))

        # Hedef klasördeki belgeleri görüntüleme butonu
        self.view_documents_button = QPushButton("Hedef Klasördeki Belgeler")
        self.view_documents_button.clicked.connect(self.view_documents)
        self.view_documents_button.setEnabled(False)  # Başlangıçta devre dışı
        self.view_documents_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))

        # Taranmış Klasörler butonu
        self.view_scanned_folders_button = QPushButton("Taranmış Klasörler")
        self.view_scanned_folders_button.clicked.connect(self.open_scanned_folders_window)
        self.view_scanned_folders_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))

        # Butonları layout'a ekleyin
        main_layout.addWidget(self.search_button)
        main_layout.addWidget(self.change_folder_button)
        main_layout.addWidget(self.view_documents_button)
        main_layout.addWidget(self.view_scanned_folders_button)

        # Ana widget'a layout'u ayarlayın
        central_widget.setLayout(main_layout)

        # Pencere başlığını ve boyutunu ayarlayın
        self.setWindowTitle("Gelişmiş Dosya Arama")
        self.resize(400, 300)
        self.setWindowIcon(QIcon('C:/VsCode Repo/Advanced File Search/icons/search.png'))


    def prompt_folder_selection(self):
        # Kullanıcıya mesaj kutusu göster
        QMessageBox.information(self, "Klasör Seçimi", "Lütfen çalışmak istediğiniz diski veya klasörü seçiniz.")

        # Kullanıcıdan klasör seçmesini isteyin
        folder = QFileDialog.getExistingDirectory(self, "Hedef Klasörü Seç")
        if folder:
            self.selected_folder = folder
            # Seçilen klasörü QSettings'e kaydedin
            settings = QSettings("AdvancedFileSearch", "App")
            settings.setValue("last_selected_folder", self.selected_folder)
            QMessageBox.information(self, "Bilgi", f"Hedef klasör seçildi: {self.selected_folder}")
           
            # İndeksleme işlemini başlat
            self.start_indexing()
        else:
            QMessageBox.warning(self, "Uyarı", "Hedef klasör seçilmedi.")
            


    def open_scanned_folders_window(self):
        self.scanned_folders_window = ScannedFoldersWindow(self)
        self.scanned_folders_window.show()
        self.hide()



    def start_indexing(self):
        # İndeksleme işlemini başlatmak için LoadingWindow'u açın
        self.loading_window = LoadingWindow(self.selected_folder)
        self.loading_window.loading_finished_signal.connect(self.on_loading_finished)
        self.loading_window.loading_error_signal.connect(self.on_loading_error)
        self.loading_window.show()
        self.hide()

    def on_loading_finished(self, indexing_performed):
        # Loading tamamlandığında butonları etkinleştir
        self.index_loaded = True
        self.search_button.setEnabled(True)
        self.view_documents_button.setEnabled(True)
        self.show()

    def on_loading_error(self, error_message):
        # Hata durumunda butonları devre dışı bırak
        self.index_loaded = False
        QMessageBox.critical(self, "Hata", f"İndeksleme sırasında bir hata oluştu: {error_message}")
        self.search_button.setEnabled(False)
        self.view_documents_button.setEnabled(False)
        self.show()

    def change_target_folder(self):
        # Kullanıcı yeni bir klasör seçmek istediğinde çalışacak fonksiyon
        folder = QFileDialog.getExistingDirectory(self, "Hedef Klasörü Seç")
        if folder:
            # Mevcut indeksleme işlemini durdurun
            if hasattr(self, 'loading_window') and self.loading_window.worker_thread.isRunning():
                self.loading_window.worker_thread.terminate()
                self.loading_window.worker_thread.wait()
                reset_globals()
            self.selected_folder = folder
            # Seçilen klasörü QSettings'e kaydedin
            settings = QSettings("AdvancedFileSearch", "App")
            settings.setValue("last_selected_folder", self.selected_folder)
            QMessageBox.information(self, "Bilgi", f"Hedef klasör değiştirildi: {self.selected_folder}")
           
            reset_globals()
            self.index_loaded = False
            # İndeksleme işlemini başlat
            self.start_indexing()
        else:
            QMessageBox.warning(self, "Uyarı", "Hedef klasör seçilmedi.")
            

    def open_search_window(self):
        self.search_window = SearchApp(self)
        self.search_window.show()
        self.hide()

    def view_documents(self):
        if not self.selected_folder:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir klasör seçin.")
            return
        self.documents_window = DocumentsWindow(self)
        self.documents_window.show()
        self.hide()
    def on_index_deleted(self, folder_name):
        selected_folder_name = os.path.basename(os.path.normpath(self.selected_folder)),
        if folder_name == selected_folder_name:
            QMessageBox.information(self, "Bilgi", "Seçili klasörün indeksi silindi. Lütfen yeni bir klasör seçin.")
            self.search_button.setEnabled(False)
            self.view_documents_button.setEnabled(False)
            self.index_loaded = False
            reset_globals()
        else:
            pass


# main bloğu
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())

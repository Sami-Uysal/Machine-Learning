from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import pandas as pd
from deney import analiz_yap
from tahmin import TahminBolumu
import sys

class DeneyBolumu(QWidget):
    modelHazir = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.veri = None
        self.model = None
        self.arayuzOlustur()

    def arayuzOlustur(self):
        layout = QVBoxLayout()

        # Veri Yükleme Butonu
        self.veri_yukle_buton = QPushButton("Veri Seti Yükle")
        self.veri_yukle_buton.clicked.connect(self.veriYukle)
        layout.addWidget(self.veri_yukle_buton)

        # Veri Önizleme
        self.veri_onizleme = QTextEdit()
        self.veri_onizleme.setReadOnly(True)
        layout.addWidget(self.veri_onizleme)

        # Deney Seçici
        self.deney_secici = QComboBox()
        self.deney_secici.addItems([
            "Orijinal Veri",
            "Dengesizlikle Başa Çıkma",
            "Normalizasyon",
            "Gürültülü Veri",
            "K-Fold Çapraz Doğrulama",
        ])
        layout.addWidget(self.deney_secici)

        # Model Seçici
        self.model_secici = QComboBox()
        self.model_secici.addItems(["Lojistik Regresyon", "Karar Ağacı", "Rastgele Orman"])
        layout.addWidget(self.model_secici)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_bar)

        # Deney Çalıştır Butonu
        self.calistir_buton = QPushButton("Deney Çalıştır")
        self.calistir_buton.clicked.connect(self.deneyCalistir)
        layout.addWidget(self.calistir_buton)

        # Sonuç Gösterimi
        self.sonuclar_gosterim = QTextEdit()
        self.sonuclar_gosterim.setReadOnly(True)
        layout.addWidget(self.sonuclar_gosterim)

        container = QWidget()
        container.setLayout(layout)
        self.setLayout(layout)

    def veriYukle(self):
        dosya_yolu, _ = QFileDialog.getOpenFileName(self, "Veri Seti Aç", "", "CSV Dosyaları (*.csv)")
        if dosya_yolu:
            self.veri = pd.read_csv(dosya_yolu, header=None)
            self.veri_onizleme.setText(str(self.veri.head()))

    def deneyCalistir(self):
        if self.veri is None:
            self.sonuclar_gosterim.setText("Lütfen önce bir veri seti yükleyin.")
            return

        deney = self.deney_secici.currentText()
        model_adi = self.model_secici.currentText()
        sonuc, model = analiz_yap(self.veri, deney, model_adi, self.progress_bar)
        self.sonuclar_gosterim.setText(sonuc)
        self.model = model
        self.modelHazir.emit(self.model)


class DeneyselArayuz(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deneysel Veri Analizi")
        self.setGeometry(100, 100, 800, 600)

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.deney_bolumu = DeneyBolumu()
        layout.addWidget(self.deney_bolumu)

        self.tahmin_bolumu = TahminBolumu()
        layout.addWidget(self.tahmin_bolumu)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.deney_bolumu.modelHazir.connect(self.tahmin_bolumu.modelAyarla)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    anaPencere = DeneyselArayuz()
    anaPencere.show()
    sys.exit(app.exec_())

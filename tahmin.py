from PyQt5.QtWidgets import *
import numpy as np

class TahminDialog(QDialog):

    def __init__(self, model, parent=None):
        super().__init__(parent)

        self.model = model
        self.veri_basliklari = [f"Özellik {i}" for i in range(1, 61)]  # 60 özellik
        self.veri_girisleri = {}

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Veri Girişi Seçenekleri
        self.giris_secim_grubu = QGroupBox("Veri Girişi Seçenekleri")
        secim_layout = QVBoxLayout()

        # Veri girişi türlerini seçme (Virgülle Ayrılmış veya Tek Tek)
        self.tek_tek_radio = QRadioButton("Tek Tek Giriş")
        self.virgul_ayrili_radio = QRadioButton("Virgülle Ayrılmış Giriş")
        self.virgul_ayrili_radio.setChecked(True)  # Varsayılan seçenek

        secim_layout.addWidget(self.tek_tek_radio)
        secim_layout.addWidget(self.virgul_ayrili_radio)

        self.giris_secim_grubu.setLayout(secim_layout)
        layout.addWidget(self.giris_secim_grubu)

        # Giriş alanı
        self.giris_grubu = QGroupBox("Tahmin İçin Veri Girişi")
        form_layout = QFormLayout()

        self.giris_text = QTextEdit()
        self.giris_text.setPlaceholderText("Verileri buraya virgülle ayırarak girin (örneğin: 0.0200, 0.0371, ..., 0.0854, ...)")
        form_layout.addRow(QLabel("Veri Girişi"), self.giris_text)

        self.giris_grubu.setLayout(form_layout)
        layout.addWidget(self.giris_grubu)

        # Eğer "Tek Tek Giriş" seçildiyse, her özellik için ayrı giriş alanları ekle
        self.tek_tek_input_fields = []
        self.tek_tek_input_area = QWidget()
        tek_tek_layout = QVBoxLayout()

        for i in range(1, 61):
            input_field = QLineEdit()
            input_field.setPlaceholderText(f"Özellik {i}")
            self.tek_tek_input_fields.append(input_field)
            tek_tek_layout.addWidget(input_field)

        self.tek_tek_input_area.setLayout(tek_tek_layout)

        # Kaydırma alanı ekle
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.tek_tek_input_area)
        self.scroll_area.setWidgetResizable(True)

        # Tek Tek Giriş alanlarını gizle
        self.setTekTekGirisVisible(False)

        layout.addWidget(self.scroll_area)

        self.tahmin_sonuc = QLabel("Tahmin Sonucu: ")
        layout.addWidget(self.tahmin_sonuc)

        self.tahmin_buton = QPushButton("Tahmin Yap")
        self.tahmin_buton.clicked.connect(self.tahminYap)
        layout.addWidget(self.tahmin_buton)

        self.setLayout(layout)

        # Seçim değiştiğinde görünümü güncelle
        self.tek_tek_radio.toggled.connect(self.onRadioButtonToggled)
        self.virgul_ayrili_radio.toggled.connect(self.onRadioButtonToggled)

    def setTekTekGirisVisible(self, visible):
        """Tek Tek Giriş alanlarının görünür olmasını kontrol eder."""
        self.scroll_area.setVisible(visible)
        self.giris_text.setVisible(not visible)

    def onRadioButtonToggled(self):
        """Seçim değiştiğinde, hangi giriş türünün görünür olacağına karar verir."""
        if self.tek_tek_radio.isChecked():
            self.setTekTekGirisVisible(True)
        elif self.virgul_ayrili_radio.isChecked():
            self.setTekTekGirisVisible(False)

    def tahminYap(self):
        if self.model is None:
            self.tahmin_sonuc.setText("Tahmin Sonucu: Önce bir model eğitin.")
            return

        # Kullanıcının seçimini alıyoruz
        if self.virgul_ayrili_radio.isChecked():
            input_text = self.giris_text.toPlainText().strip()
            if not input_text:
                self.tahmin_sonuc.setText("Tahmin Sonucu: Lütfen veri girin.")
                return

            try:
                # Girdiği veriyi virgüllere göre ayırıyoruz ve gereksiz boşlukları temizliyoruz
                input_values = [value.strip() for value in input_text.split(",")]

                # 60 özellik olmalı (sayısal veriler)
                if len(input_values) != 60:
                    self.tahmin_sonuc.setText(f"Tahmin Sonucu: Hatalı giriş. 60 veri girilmelidir.")
                    return

                # Özellikleri (ilk 60 değeri) alıyoruz
                features = [float(value) for value in input_values]  # İlk 60 değer özellikler

            except ValueError:
                self.tahmin_sonuc.setText("Tahmin Sonucu: Lütfen verileri doğru formatta girin (örneğin: 0.0200, 0.0371, ..., 0.0854).")
                return

        elif self.tek_tek_radio.isChecked():
            features = []
            for input_field in self.tek_tek_input_fields:
                try:
                    value = float(input_field.text().strip())
                    features.append(value)
                except ValueError:
                    self.tahmin_sonuc.setText("Tahmin Sonucu: Lütfen her özelliği doğru formatta girin.")
                    return

            if len(features) != 60:
                self.tahmin_sonuc.setText("Tahmin Sonucu: 60 özellik girilmelidir.")
                return

        # Girdi verisini numpy dizisine dönüştürüyoruz
        girdi = np.array(features).reshape(1, -1)

        # Modeli kullanarak tahmin yapıyoruz
        tahmin = self.model.predict(girdi)

        # Modelin tahminini kontrol ediyoruz (R ya da Maden)
        if tahmin[0] == 'R':
            sonuc = "Bu Nesne Taş"
        else:
            sonuc = "Bu Nesne Maden"

        self.tahmin_sonuc.setText(f"Tahmin Sonucu: {sonuc}")


class TahminBolumu(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.model = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.tahmin_buton = QPushButton("Tahmin Yap")
        self.tahmin_buton.clicked.connect(self.showPredictionDialog)
        layout.addWidget(self.tahmin_buton)

        self.setLayout(layout)

    def modelAyarla(self, model):
        self.model = model

    def showPredictionDialog(self):
        if self.model is None:
            QMessageBox.warning(self, "Model Eksik", "Lütfen önce bir model eğitin.")
            return

        self.dialog = TahminDialog(self.model, self)
        self.dialog.exec_()

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score, precision_score, f1_score)
import time

MODELLER = {
    "Lojistik Regresyon": LogisticRegression(max_iter=1000),
    "Karar Ağacı": DecisionTreeClassifier(),
    "Rastgele Orman": RandomForestClassifier(),
}

def analiz_yap(veri, deney, model_adi, progress_bar):
    model = MODELLER[model_adi]
    X = veri.drop(columns=veri.columns[-1], axis=1)
    Y = veri[veri.columns[-1]]

    if deney == "Dengesizlikle Başa Çıkma":
        X, Y = resample(X, Y, stratify=Y, replace=True)

    elif deney == "Normalizasyon":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    elif deney == "Gürültülü Veri":
        gürültü = np.random.normal(0, 0.1, X.shape)
        X += gürültü

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

    if deney == "K-Fold Çapraz Doğrulama":
        skorlar = cross_val_score(model, X, Y, cv=5)
        return f"K-Fold Çapraz Doğrulama Doğruluğu: {skorlar.mean():.4f}", None

    for i in range(1, 101):
        time.sleep(0.02)
        progress_bar.setValue(i)

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    dogruluk = accuracy_score(Y_test, Y_pred)
    duyarlilik = recall_score(Y_test, Y_pred, pos_label=Y.unique()[0])
    kesinlik = precision_score(Y_test, Y_pred, pos_label=Y.unique()[0])
    f1 = f1_score(Y_test, Y_pred, pos_label=Y.unique()[0])
    karisiklik_matrisi = confusion_matrix(Y_test, Y_pred)

    sonuclar = (
        f"Doğruluk: {dogruluk:.4f}\n"
        f"Duyarlılık: {duyarlilik:.4f}\n"
        f"Kesinlik: {kesinlik:.4f}\n"
        f"F1-Skor: {f1:.4f}\n"
        f"Karmaşıklık Matrisi:\n{karisiklik_matrisi}"
    )
    return sonuclar, model


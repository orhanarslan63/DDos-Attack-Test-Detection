import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
# <<<--- confusion_matrix import'u zaten vardı, sadece emin olalım ---<<<
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import sys # Hata durumunda çıkmak için

# --- YENİ VERİ SETİ YOLU (GÜNCELLE!) ---
CSV_PATH = "C:/Users/asd/Desktop/dataset_combined_sampled.csv" # <<<--- LÜTFEN KONTROL EDİN/GÜNCELLEYİN

# --- YENİ KAYIT DOSYA İSİMLERİ ---
NEW_MODEL_FILENAME = "ddos_model_v3.keras"
NEW_SCALER_FILENAME = "ddos_scaler_v3.pkl"

# --- Hedef Değişken Sütun Adı ---
TARGET_COLUMN = 'tcp.time_delta' # <<<--- CSV'deki GERÇEK etiket sütunu adını yazın!

# --- Seçilen Özellik Sütunları (14 adet) ---
feature_cols = [
    'tcp.srcport', 'tcp.dstport', 'ip.proto', 'frame.len',
    'tcp.flags.syn', 'tcp.flags.ack', 'tcp.flags.push',
    'tcp.flags.reset', 'tcp.flags.fin', 'ip.flags.df',
    # 'ip.flags.mf', # Çıkarıldı
    'ip.frag_offset', 'ip.ttl', 'tcp.window_size', 'tcp.time_delta'
]

# === Kodun Geri Kalanı (Yükleme, Ön İşleme, Sayısal Dönüşüm) ===

# CSV dosyasını okuyun
try:
    df = pd.read_csv(CSV_PATH)
    print(f"'{CSV_PATH}' başarıyla okundu.")
except FileNotFoundError:
    print(f"HATA: CSV dosyası bulunamadı: {CSV_PATH}"); sys.exit(1)
except Exception as e:
    print(f"HATA: CSV dosyası okunurken hata: {e}"); sys.exit(1)

print("Veri seti örneği:\n", df.head())
print("\nVeri seti bilgileri:"); df.info(); print("-" * 30)

# Hedef değişken kontrolü
if TARGET_COLUMN not in df.columns:
    print(f"HATA: Hedef değişken '{TARGET_COLUMN}' sütunu bulunamadı!")
    print(f"CSV sütunları: {df.columns.tolist()}"); sys.exit(1)

# Özellik sütunlarının varlığını kontrol et
missing_cols = [col for col in feature_cols if col not in df.columns]
if missing_cols:
    print(f"HATA: Veri setinde şu gerekli sütunlar eksik: {missing_cols}")
    print(f"CSV sütunları: {df.columns.tolist()}"); sys.exit(1)

print(f"\nModel için kullanılacak özellikler ({len(feature_cols)} adet): {feature_cols}")

# Genel NaN doldurma (isteğe bağlı, sayısal dönüşüm sonrası da yapılabilir)
# df[feature_cols] = df[feature_cols].fillna(0)
# print("Genel eksik veriler (varsa) 0 ile dolduruldu.")

# Özellik (X) ve Etiket (y) belirleme
X = df[feature_cols].copy()
y = df[TARGET_COLUMN]

print("\nÖzelliklerin ilk veri tipleri kontrol ediliyor...")
print(X.dtypes)

# Object tipindeki sütunları sayısal yapma
print("\nObject tipindeki sütunları sayısal yapma girişimi...")
cols_to_convert = ['ip.proto', 'frame.len', 'tcp.time_delta']
conversion_issues = False
for col in cols_to_convert:
    if col in X.columns:
        if X[col].dtype == 'object':
            print(f"'{col}' sütunu pd.to_numeric ile dönüştürülüyor (hatalar NaN yapılacak)...")
            original_dtype = X[col].dtype
            X[col] = pd.to_numeric(X[col], errors='coerce')
            nan_count = X[col].isna().sum()
            if nan_count > 0:
                print(f"'{col}' sütununda {nan_count} adet NaN bulundu, 0 ile dolduruluyor.")
                X[col] = X[col].fillna(0) # <<<--- NaN Doldurma
            try:
                 if X[col].fillna(0).apply(float.is_integer).all(): X[col] = X[col].astype(np.int64)
            except Exception: pass
            print(f"'{col}' sütunu: Orjinal tip={original_dtype}, Yeni tip={X[col].dtype}")
        else:
             print(f"'{col}' sütunu zaten sayısal ({X[col].dtype}), dönüştürme atlanıyor.")
    else:
        print(f"UYARI: Dönüştürülecek '{col}' sütunu X DataFrame'inde bulunamadı.")
        conversion_issues = True

# Sayısal dönüşüm sonrası kalan NaN'ları da dolduralım (varsa)
final_nan_counts = X.isna().sum()
if final_nan_counts.sum() > 0:
    print("\nSayısal dönüşüm sonrası kalan NaN değerler 0 ile dolduruluyor...")
    print(final_nan_counts[final_nan_counts > 0])
    X = X.fillna(0)
else:
    print("\nSayısal dönüşüm sonrası NaN değer bulunmuyor.")


# Son kontrol: Hala object kalan var mı?
final_dtypes = X[feature_cols].dtypes
if (final_dtypes == 'object').any():
     print("\n!!! HATA: Sayısal dönüşüm sonrası hala 'object' tipinde sütunlar var!")
     print(final_dtypes[final_dtypes == 'object'])
     sys.exit("Veri tipi sorunu devam ediyor.")
else:
     print("\nTüm özellik sütunları başarıyla sayısal tipe dönüştürüldü.")
print("--- Sayısallaştırma Sonu ---\n")

# Etiket sütununu sayısal yapma
if y.dtype == 'object':
    le_label = LabelEncoder()
    y = le_label.fit_transform(y)
    print(f"'{TARGET_COLUMN}' sütunu LabelEncoder ile dönüştürüldü.")
    print("Etiket Sınıfları:", le_label.classes_)
    num_classes = len(le_label.classes_)
else:
    num_classes = y.nunique()
    print(f"'{TARGET_COLUMN}' sütunu zaten sayısal. Sınıf sayısı: {num_classes}")

if num_classes < 2: print("Hata: Yetersiz sınıf sayısı."); sys.exit(1)

# Veriyi Eğitim ve Test Setlerine Ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"\nEğitim seti boyutu: {X_train.shape}"); print(f"Test seti boyutu: {X_test.shape}")

# Özellik Ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Özellikler StandardScaler ile ölçeklendirildi.")

# --- Derin Öğrenme Modeli ---
input_dim = X_train_scaled.shape[1]
print(f"Model girdi boyutu (özellik sayısı): {input_dim}")

if num_classes == 2: output_units=1; output_activation='sigmoid'; loss_function='binary_crossentropy'
elif num_classes > 2: output_units=num_classes; output_activation='softmax'; loss_function='sparse_categorical_crossentropy'
else: print("Hata: Geçersiz sınıf sayısı."); sys.exit(1)

model = Sequential(name="DeepNetworkTrafficClassifier_v3")
model.add(Input(shape=(input_dim,), name="InputLayer"))
model.add(Dense(64, activation='relu', name="HiddenLayer1"))
model.add(Dropout(0.3, name="Dropout1"))
model.add(Dense(32, activation='relu', name="HiddenLayer2"))
model.add(Dropout(0.3, name="Dropout2"))
model.add(Dense(output_units, activation=output_activation, name="OutputLayer"))

print("\nModel Mimarisi:"); model.summary(line_length=100)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
print("Model derlendi.")

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
print("EarlyStopping ayarlandı.")

print("\nModel Eğitimi Başlatılıyor...")
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=64,
                    validation_split=0.2, callbacks=[early_stopping], verbose=1)
print("Model Eğitimi Tamamlandı.")

# --- Model Değerlendirme ---
print("\n--- Model Değerlendirme ---")
# Keras evaluate
loss, accuracy_keras = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nKeras Test Kaybı: {loss:.4f}, Doğruluk: {accuracy_keras:.4f}")

# Sklearn metrikleri için tahmin yap
if output_activation == 'sigmoid':
    y_pred_proba = model.predict(X_test_scaled)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
else: # softmax
    y_pred_proba = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_proba, axis=1)

# Metrikleri hesapla
accuracy_sklearn = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# <<<--- DEĞİŞİKLİK: cm değişkeni burada atanıyor ---<<<
cm = confusion_matrix(y_test, y_pred)
# <<<--- DEĞİŞİKLİK SONU ---<<<

print("\nScikit-learn Metrikleri:")
print(f"  Accuracy: {accuracy_sklearn:.4f}")
print(f"  Precision (Weighted): {precision:.4f}")
print(f"  Recall (Weighted): {recall:.4f}")
print(f"  F1 Score (Weighted): {f1:.4f}")
print("  Confusion Matrix:\n", cm) # Şimdi cm değişkenini kullanıyoruz

# Confusion Matrix Görselleştirme
try:
    plt.figure(figsize=(8, 6))
    if 'le_label' in locals() and hasattr(le_label, 'classes_'):
        labels = le_label.classes_
    else:
        # y_test ve y_pred içindeki tüm benzersiz etiketleri al
        labels = sorted(np.unique(np.concatenate((y_test, y_pred))))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=[f'Predicted {lbl}' for lbl in labels],
                yticklabels=[f'Actual {lbl}' for lbl in labels])
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.title('Confusion Matrix (Model v3)')
    plt.show()
except Exception as e:
     print(f"\nUYARI: Confusion matrix çizilemedi: {e}")


# Eğitim Geçmişini Görselleştirme
try:
    plt.figure(figsize=(12, 5))
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy'); plt.ylabel('Accuracy'); plt.xlabel('Epoch'); plt.legend(loc='lower right')
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss'); plt.ylabel('Loss'); plt.xlabel('Epoch'); plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
except Exception as e:
     print(f"\nUYARI: Eğitim grafikleri çizilemedi: {e}")


# --- Modeli ve Scaler'ı Kaydetme ---
# Bu satırlar artık NameError olmadan çalışmalı
try:
    model.save(NEW_MODEL_FILENAME)
    print(f"\nModel '{NEW_MODEL_FILENAME}' olarak başarıyla kaydedildi.")
except Exception as e:
    print(f"\nHATA: Model kaydedilemedi: {e}")

try:
    with open(NEW_SCALER_FILENAME, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler '{NEW_SCALER_FILENAME}' olarak başarıyla kaydedildi.")
except Exception as e:
     print(f"\nHATA: Scaler kaydedilemedi: {e}")

print("\nEğitim betiği tamamlandı.")
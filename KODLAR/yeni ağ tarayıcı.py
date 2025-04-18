import pyshark
import numpy as np
from datetime import datetime
import joblib
import sys
import time
import warnings
import tensorflow as tf
# Gerekli pyshark exception'ını doğru yerden import edelim
from pyshark.tshark.tshark import TSharkNotFoundException
# scikit-learn kaynaklı bazı uyarıları bastırmak isteyebilirsin
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')

# --- YAPILANDIRMA (v3 DOSYA ADLARI VE 14 ÖZELLİK) ---
# <<<--- DEĞİŞTİRİLDİ: v3 model/scaler yolları ---
MODEL_PATH = 'ddos_model_v3.keras'
SCALER_PATH = 'ddos_scaler_v3.pkl'
# --- v3 yolları sonu ---

INTERFACE = '4'  # <<<--- DOĞRU ARAYÜZ NUMARASINI VEYA ADINI KULLANIN ---

# <<<--- DEĞİŞTİRİLDİ: Özellik sırası (14 özellik - ip.flags.mf çıkarıldı) ---
FEATURE_ORDER = [
    'tcp.srcport',
    'tcp.dstport',
    'ip.proto',
    'frame.len',
    'tcp.flags.syn',
    'tcp.flags.ack', # ACK Flag
    'tcp.flags.push',
    'tcp.flags.reset',
    'tcp.flags.fin',
    'ip.flags.df',
    # 'ip.flags.mf', # <<<--- BU SATIR ÇIKARILDI ---
    'ip.frag_offset',
    'ip.ttl',
    'tcp.window_size',
    'tcp.time_delta'
]
NUM_FEATURES = len(FEATURE_ORDER) # Otomatik olarak 14 olacak
# --- Yeni özellik sırası sonu ---

# --- YARDIMCI FONKSİYON: GÜVENLİ BAYRAK DÖNÜŞÜMÜ (Aynı) ---
def safe_flag_to_int(flag_value):
    if isinstance(flag_value, bool): return 1 if flag_value else 0
    elif isinstance(flag_value, str):
        if flag_value == '1' or flag_value.lower() == 'true': return 1
        elif flag_value == '0' or flag_value.lower() == 'false': return 0
        else: print(f"--- UYARI: Beklenmedik bayrak değeri '{flag_value}', 0."); return 0
    elif isinstance(flag_value, int): return 1 if flag_value == 1 else 0
    else: print(f"--- UYARI: Beklenmedik bayrak tipi '{type(flag_value)}', 0."); return 0

# --- MODEL VE SCALER YÜKLEME (Yolları yukarıda güncellendi) ---
print("Model ve Scaler yükleniyor...")
model = None
scaler = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Keras Modeli başarıyla yüklendi: {MODEL_PATH}")
    model.summary(line_length=100) # Özetin daha iyi görünmesi için
except FileNotFoundError: print(f"HATA: Model dosyası bulunamadı: {MODEL_PATH}"); sys.exit(1)
except Exception as e: print(f"HATA: Keras Modeli yüklenemedi ({MODEL_PATH}): {e}"); sys.exit(1)

try:
    scaler = joblib.load(SCALER_PATH)
    print(f"Scaler başarıyla yüklendi: {SCALER_PATH}")
    if not hasattr(scaler, 'transform'): print(f"HATA: Yüklenen nesne ({SCALER_PATH}) scaler değil!"); sys.exit(1)

    # Scaler özellik sayısı kontrolü
    expected_features = 'Bilinmiyor'
    try:
        if hasattr(scaler, 'n_features_in_'): expected_features = scaler.n_features_in_
        elif hasattr(scaler, 'scale_'): expected_features = len(scaler.scale_)
        print(f"--- Scaler beklenen özellik sayısı: {expected_features}")
        if isinstance(expected_features, int) and expected_features != NUM_FEATURES:
            print(f"!!! UYARI: Scaler {expected_features} özellik bekliyor, kod {NUM_FEATURES} özellik çıkaracak.")
    except Exception as e: print(f"--- Scaler özellik sayısı kontrol edilemedi: {e}")

except FileNotFoundError: print(f"HATA: Scaler dosyası bulunamadı: {SCALER_PATH}"); sys.exit(1)
except Exception as e: print(f"HATA: Scaler yüklenemedi ({SCALER_PATH}): {e}"); sys.exit(1)
print("------------------------------------\n")

# --- ÖZELLİK ÇIKARMA FONKSİYONU (14 ÖZELLİK İÇİN GÜNCELLENDİ) ---
# <<<--- DEĞİŞTİRİLDİ: ip.flags.mf çıkarıldı ---
def extract_features(packet, packet_num_for_debug):
    """Verilen paketten 14 özelliği çıkarır."""
    features = {}
    try:
        if not hasattr(packet, 'ip') or not hasattr(packet, 'tcp'): return None

        features['tcp.srcport'] = int(getattr(packet.tcp, 'srcport', 0))
        features['tcp.dstport'] = int(getattr(packet.tcp, 'dstport', 0))
        features['ip.proto'] = int(getattr(packet.ip, 'proto', 0))
        features['frame.len'] = int(getattr(packet.frame_info, 'len', 0))

        features['tcp.flags.syn'] = safe_flag_to_int(getattr(packet.tcp, 'flags_syn', '0'))
        features['tcp.flags.ack'] = safe_flag_to_int(getattr(packet.tcp, 'flags_ack', '0'))
        features['tcp.flags.push'] = safe_flag_to_int(getattr(packet.tcp, 'flags_push', '0'))
        features['tcp.flags.reset'] = safe_flag_to_int(getattr(packet.tcp, 'flags_rst', '0'))
        features['tcp.flags.fin'] = safe_flag_to_int(getattr(packet.tcp, 'flags_fin', '0'))

        features['ip.flags.df'] = safe_flag_to_int(getattr(packet.ip, 'flags_df', '0'))
        # features['ip.flags.mf'] = safe_flag_to_int(getattr(packet.ip, 'flags_mf', '0')) # <<<--- ÇIKARILDI ---

        features['ip.frag_offset'] = int(getattr(packet.ip, 'frag_offset', 0))
        features['ip.ttl'] = int(getattr(packet.ip, 'ttl', 0))
        win_size = getattr(packet.tcp, 'window_size_value', getattr(packet.tcp, 'window_size', 0))
        features['tcp.window_size'] = int(win_size)
        features['tcp.time_delta'] = float(getattr(packet.tcp, 'time_delta', 0.0))

        # Özellikleri FEATURE_ORDER sırasına göre diz
        ordered_feature_list = []
        for f_name in FEATURE_ORDER:
            val = features.get(f_name)
            if val is None:
                 # Bu durumun oluşmaması lazım ama kontrol olarak kalsın
                 print(f"--- UYARI: Paket {packet_num_for_debug} için özellik '{f_name}'=None. Atlanıyor.")
                 return None
            ordered_feature_list.append(val)

        # Son kontrol: Liste uzunluğu doğru mu?
        if len(ordered_feature_list) != NUM_FEATURES:
            print(f"--- HATA: Paket {packet_num_for_debug} için çıkarılan özellik sayısı yanlış! Beklenen:{NUM_FEATURES}, Çıkarılan:{len(ordered_feature_list)}")
            return None

        return ordered_feature_list

    except Exception as e:
        # print(f"--- HATA: Paket {packet_num_for_debug} için özellik çıkarılamadı: {e}") # Detaylı debug
        return None
# --- extract_features sonu ---

# --- ÖN İŞLEME FONKSİYONU (Aynı) ---
def preprocess_features(feature_list, scaler_obj, packet_num_for_debug):
    """Özellik listesini yüklenen scaler ile ön işler."""
    try:
        features_np = np.array(feature_list).reshape(1, -1)
        # Şekil kontrolü
        if features_np.shape[1] != NUM_FEATURES:
            print(f"HATA: Paket {packet_num_for_debug} ön işleme sırasında özellik sayısı uyumsuz! Beklenen: {NUM_FEATURES}, Gelen: {features_np.shape[1]}")
            return None
        scaled_features = scaler_obj.transform(features_np)
        return scaled_features
    except Exception as e:
        print(f"HATA: Paket {packet_num_for_debug} ön işleme sırasında hata: {e}")
        return None

# --- PAKET İŞLEME CALLBACK FONKSİYONU (Aynı) ---
packet_count = 0
prediction_count = 0
start_time = time.time()

def process_packet(packet):
    """Her yakalanan paket için çağrılacak fonksiyon."""
    global packet_count, prediction_count, start_time
    packet_count += 1

    extracted_data = extract_features(packet, packet_count)

    if extracted_data:
        processed_data = preprocess_features(extracted_data, scaler, packet_count)

        if processed_data is not None:
            try:
                prediction_probabilities = model.predict(processed_data)

                # Eğitimde kullanılan çıktı katmanına göre yorumlama
                if prediction_probabilities.shape[1] == 1: # Sigmoid varsayımı
                    probability = prediction_probabilities[0][0]
                    prediction_label = 1 if probability > 0.5 else 0
                    interpretation = "Saldiri" if prediction_label == 1 else "Normal" # <<<--- KENDİ ETİKETLERİNİZE GÖRE GÜNCELLEYİN
                else: # Softmax varsayımı
                    prediction_label = np.argmax(prediction_probabilities[0])
                    interpretation = f"Sinif {prediction_label}" # <<<--- KENDİ ETİKETLERİNİZE GÖRE GÜNCELLEYİN

                prediction_count += 1
                print(f"Paket {packet_count}: {packet.ip.src}:{packet.tcp.srcport} -> {packet.ip.dst}:{packet.tcp.dstport} | Tahmin: {prediction_label} ({interpretation})")

            except Exception as e:
                print(f"HATA: Paket {packet_count} için tahmin sırasında hata: {e}")

    # Durum Güncellemesi
    if packet_count % 200 == 0:
        elapsed_time = time.time() - start_time
        rate = packet_count / elapsed_time if elapsed_time > 0 else 0
        pred_rate = prediction_count / elapsed_time if elapsed_time > 0 else 0
        print(f"\n--- Durum [{datetime.now().strftime('%H:%M:%S')}] ---")
        print(f"   İşlenen Paket: {packet_count} ({rate:.2f} pkt/s)")
        print(f"   Yapılan Tahmin: {prediction_count} ({pred_rate:.2f} pred/s)")
        print(f"   Geçen Süre: {elapsed_time:.2f} s\n")


# --- YAKALAMAYI BAŞLAT (Güncel Hata Yakalama ile) ---
print("--------------------------------------------------")
print(f"Gerçek zamanlı paket yakalama başlatılıyor...")
print(f"Arayüz: {INTERFACE or 'Varsayılan'}")
print(f"Kullanılan Model: {MODEL_PATH}")
print(f"Kullanılan Scaler: {SCALER_PATH}")
print(f"Kullanılan Özellik Sayısı: {NUM_FEATURES}")
print("Durdurmak için Ctrl+C tuşlarına basın.")
print("--------------------------------------------------")

capture = None
try:
    capture = pyshark.LiveCapture(
        interface=INTERFACE,
        bpf_filter='tcp or ip' # Sadece IP veya TCP paketleri
        # tshark_path='C:/Program Files/Wireshark/tshark.exe' # Gerekirse tshark yolu
    )
    capture.apply_on_packets(process_packet, timeout=None)

# <<<--- Hata Yakalama Blokları (Güncel) ---<<<
except PermissionError: print("\nHATA: İzin Hatası! Yönetici olarak çalıştırın.")
except ImportError as e: print(f"\nHATA: Kütüphane bulunamadı: {e}. Kurulumları kontrol edin.")
except TSharkNotFoundException: print("\nHATA: TShark bulunamadı! Wireshark kurulu mu ve PATH'de mi?")
except UnicodeDecodeError as e: print(f"\nHATA: TShark kodlama hatası: {e}. Pyshark güncelleyin/düzenleyin.")
except Exception as e: print(f"\nHATA: Yakalama sırasında beklenmedik hata: {e}. Arayüz: '{INTERFACE}' doğru mu?"); print(type(e)) # Hatanın tipini de yazdır
except KeyboardInterrupt: print("\n--- Durduruldu ---")
# <<<----------------------------------------<<<
finally:
    if capture and hasattr(capture, 'close'): capture.close()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\n--------------------------------------------------")
    print(f"Toplam {packet_count} paket işlendi.")
    print(f"Toplam {prediction_count} tahmin yapıldı.")
    print(f"Toplam süre: {elapsed_time:.2f} saniye.")
    print("Program sonlandırılıyor.")
    print("--------------------------------------------------")
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import roc_curve, auc
import soundfile as sf

# ---------------------- CONSTANTS ----------------------
MODEL_NAME = "model_ours_cnn2D_logmel_20251219_1629.h5"
DATASET_PATH = "dataset"
SAMPLE_RATE = 16000
AUDIO_LENGTH = 2

# ---------------------- FEATURE EXTRACTION ----------------------
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64

def extract_mel_spectrogram_tf(audio):
    # audio: 1D float32 tensor, shape (TARGET_SAMPLES,)
    # 1) STFT
    stft = tf.signal.stft(
        audio,
        frame_length=N_FFT,
        frame_step=HOP_LENGTH,
        fft_length=N_FFT,
        window_fn=tf.signal.hann_window,
        pad_end=False
    )
    # 2) Magnitude
    magnitude = tf.abs(stft)
    # 3) Mel filterbank
    mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=N_FFT // 2 + 1,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=0.0,
        upper_edge_hertz=SAMPLE_RATE / 2
    )
    mel_spec = tf.matmul(magnitude, mel_filterbank)
    # 4) Log compression (numerical stability only)
    mel_spec = tf.math.log(mel_spec + 1e-6)
    # 5) Channel dimension for CNN
    mel_spec = tf.expand_dims(mel_spec, axis=-1)

    return mel_spec



# ---------------------- LOAD AUDIO DATA ----------------------
def load_audio_data(folder_path, label):
    data, labels, filenames = [], [], []
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

    for file in tqdm(files, desc=f"Loading {folder_path}", unit="file"):
        path = os.path.join(folder_path, file)

        audio, sr = sf.read(path, dtype='int16')
        audio = audio.astype(np.float32) / 32768.0

        target_len = SAMPLE_RATE * AUDIO_LENGTH
        if len(audio) > target_len:
            audio = audio[:target_len]
        elif len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))

        audio_tf = tf.convert_to_tensor(audio, dtype=tf.float32)
        mel_spec = extract_mel_spectrogram_tf(audio_tf).numpy()

        data.append(mel_spec)
        labels.append(label)
        filenames.append(file)

    return np.array(data), np.array(labels), np.array(filenames)


# ---------------------- LOAD TEST DATA ----------------------
print("Loading test dataset...\n")

X_drone, y_drone, files_drone = load_audio_data(os.path.join(DATASET_PATH, "test", "drone"), label=1)
X_bg, y_bg, files_bg = load_audio_data(os.path.join(DATASET_PATH, "test", "background"), label=0)

X_test = np.concatenate([X_drone, X_bg])
y_test = np.concatenate([y_drone, y_bg])
filenames = np.concatenate([files_drone, files_bg])

print(f"\nFinal test set shape: {X_test.shape}")


# ---------------------- LOAD MODEL ----------------------
model = load_model(MODEL_NAME)
print("Model loaded.")

# ---------------------- PREDICT ----------------------
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

# ---------------------- REPORTS ----------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Background", "Drone"]))

# ---------------------- Confusion Matrix ----------------------
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Background", "Drone"],
            yticklabels=["Background", "Drone"])
plt.title("Confusion Matrix – Test Data")
plt.xlabel("Predicted")
plt.ylabel("True")
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
plt.savefig(f"testing_conf_matrix_{timestamp}.png", dpi=300)
plt.show()


# ---------------------- ROC CURVE ----------------------
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--')  # diagonal line for random classifier
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Drone Detection Model")
plt.legend(loc="lower right")

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
plt.savefig(f"ROC_curve_{timestamp}.png", dpi=300)
plt.show()

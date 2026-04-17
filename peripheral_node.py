import os
import numpy as np
import tensorflow as tf
import tflite_runtime.interpreter as tflite
import pyaudio
import psutil
import soundfile as sf
import time
import threading
import json
from zoneinfo import ZoneInfo
from datetime import datetime
athens_tz = ZoneInfo("Europe/Athens")
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from lora import LoRaTransmitter

############################################################################################################
################################                  CONSTANTS                 ################################
############################################################################################################
node_id = 2

SLIDING_SPEED = 1                 # seconds per prediction step
CONF_HISTORY_LEN = 50

# --- AUDIO CONFIG ---
RAW_CHANNELS = 6                  # ReSpeaker raw outputs
SAMPLE_RATE = 16000
AUDIO_LENGTH = 2                  # seconds
TARGET_SAMPLES = SAMPLE_RATE * AUDIO_LENGTH  # 32000

CHUNK = 1024
STEP_SIZE = int(SLIDING_SPEED * SAMPLE_RATE)

# Mel-spectrogram params
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64

# AoA / GCC-PHAT params
C_SOUND = 343.0
MIC_RADIUS_M = 0.0271

PHAT_INTERP = 16
AOA_FRAME = 2048                  # samples per mic used per AoA estimate
AOA_UPDATE_SEC = 0.05             # AoA update interval
AOA_SMOOTH_N = 7
AOA_MIN_PEAK_RATIO = 1.2

############################################################################################################
################################          SHARED STATE / BUFFERS            ################################
############################################################################################################
# Prediction ring buffer (mono merged int16)
pred_buffer = np.zeros(TARGET_SAMPLES, dtype=np.int16)
pred_lock = threading.Lock()

# AoA ring buffer (store 4 mics as [N,4] int16)
AOA_RING_LEN = max(AOA_FRAME * 8, 16384)   # plenty of history for slicing
aoa_buffer = np.zeros((AOA_RING_LEN, 4), dtype=np.int16)
aoa_write_idx = 0
aoa_lock = threading.Lock()

# Shared outputs
latest_probability = 0.0
confidence_history = []
latest_angle = -1
angle_lock = threading.Lock()

stop_event = threading.Event()

############################################################################################################
################################                LOAD MODEL                  ################################
############################################################################################################
try:
    interpreter = tflite.Interpreter(
        model_path="models/model_ours_cnn2D_logmel_20251219_1629.tflite"
    )
    interpreter.allocate_tensors()
    print("\033[96m[Model] TFLite Mel-spectrogram model loaded successfully.\033[0m")
    print("[Model] Input details:", interpreter.get_input_details())
except Exception as e:
    print("\033[91m[Model] Failed to load the TFLite model:\033[0m", e)
    raise SystemExit(1)

############################################################################################################
################################          INITIALIZE AUDIO STREAM           ################################
############################################################################################################
p = pyaudio.PyAudio()
device_index = None
respeaker_keyword = "respeaker"

print("\033[96m[Microphone] Checking for ReSpeaker microphone device:\033[0m")
for i in range(p.get_device_count()):
    dev_info = p.get_device_info_by_index(i)
    name_lower = dev_info["name"].lower()
    if dev_info["maxInputChannels"] > 0 and respeaker_keyword in name_lower:
        device_index = i
        print(f"\033[96m[Microphone] Found ReSpeaker device: index {i}, name: {dev_info['name']}\033[0m")
        break

if device_index is None:
    print("\033[91m[Microphone] Error: ReSpeaker microphone not found. Please check that it is connected.\033[0m")
    p.terminate()
    raise SystemExit(1)

print(f"\033[96m[Microphone] Using device index: {device_index}\033[0m")

stream = p.open(
    format=pyaudio.paInt16,
    channels=RAW_CHANNELS,
    rate=SAMPLE_RATE,
    input=True,
    input_device_index=device_index,
    frames_per_buffer=CHUNK,
)
print("\033[96m[Microphone] Audio stream opened (6 channels, int16). Starting capture thread...\033[0m")  


############################################################################################################
################################             LORA SENDER INIT               ################################
############################################################################################################
lora = LoRaTransmitter()
lora.setup_module()



############################################################################################################
################################             SOUND CAPTURE THREAD           ################################
############################################################################################################
def audio_capture_loop():
    """
    The ONLY thread that calls stream.read().
    It updates:
      - pred_buffer: mono merged (mics 1-4 summed) sliding window of TARGET_SAMPLES
      - aoa_buffer: ring buffer of last AOA_RING_LEN samples for 4 mics (1-4)
    """
    global pred_buffer, aoa_buffer, aoa_write_idx

    while not stop_event.is_set():
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            multi = np.frombuffer(data, dtype=np.int16).astype(np.int32)

            # Extract mics 1–4 (same indexing you used)
            mic1 = multi[1::RAW_CHANNELS]
            mic2 = multi[2::RAW_CHANNELS]
            mic3 = multi[3::RAW_CHANNELS]
            mic4 = multi[4::RAW_CHANNELS]

            # Build AoA frame [CHUNK,4]
            mics_4 = np.stack([
                mic1.astype(np.int16),
                mic2.astype(np.int16),
                mic3.astype(np.int16),
                mic4.astype(np.int16),
            ], axis=1)  # (CHUNK,4)

            # Merge mono for prediction (sum mics 1–4)
            merged = (mic1 + mic2 + mic3 + mic4) // 4
            merged = np.clip(merged, -32768, 32767).astype(np.int16)

            # Update prediction sliding buffer
            with pred_lock:
                shift = merged.shape[0]
                if shift >= TARGET_SAMPLES:
                    pred_buffer[:] = merged[-TARGET_SAMPLES:]
                else:
                    pred_buffer[:-shift] = pred_buffer[shift:]
                    pred_buffer[-shift:] = merged


            # Update AoA ring buffer
            with aoa_lock:
                n = mics_4.shape[0]
                end = aoa_write_idx + n

                if end <= AOA_RING_LEN:
                    aoa_buffer[aoa_write_idx:end, :] = mics_4
                else:
                    first = AOA_RING_LEN - aoa_write_idx
                    aoa_buffer[aoa_write_idx:, :] = mics_4[:first, :]
                    remain = n - first
                    aoa_buffer[:remain, :] = mics_4[first:, :]

                aoa_write_idx = (aoa_write_idx + n) % AOA_RING_LEN

        except Exception as e:
            print(f"\033[91m[CAPTURE LOOP ERROR] {e}\033[0m")
            time.sleep(0.05)



############################################################################################################
################################           AI PREDICTIONS SECTION           ################################
############################################################################################################

def extract_logmel_tf(audio_1d_float32: tf.Tensor) -> tf.Tensor:
    """
    audio_1d_float32: tf.float32 tensor, shape (TARGET_SAMPLES,)
    returns: tf.float32 tensor, shape (T, N_MELS, 1)
    """
    stft = tf.signal.stft(
        audio_1d_float32,
        frame_length=N_FFT,
        frame_step=HOP_LENGTH,
        fft_length=N_FFT,
        window_fn=tf.signal.hann_window,
        pad_end=False 
    )

    magnitude = tf.abs(stft) 

    mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=N_FFT // 2 + 1,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=0.0,
        upper_edge_hertz=SAMPLE_RATE / 2
    )

    mel_spec = tf.matmul(magnitude, mel_filterbank)
    mel_spec = tf.math.log(mel_spec + 1e-6) 
    mel_spec = tf.expand_dims(mel_spec, axis=-1)  # (T, 64, 1)
    return mel_spec


def preprocess_audio_training(audio_int16_or_float: np.ndarray) -> np.ndarray:
    """
    Accepts:
      - mono int16 array shape (N,)  OR
      - mono float array shape (N,)
    Produces:
      - model input float32 array shape (1, T, 64, 1)
    """
    audio = np.asarray(audio_int16_or_float)

    # 1) Ensure float32 in [-1, 1] exactly like training
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)

    # 2) Force exact 2 seconds (32000 samples)
    if audio.shape[0] > TARGET_SAMPLES:
        audio = audio[:TARGET_SAMPLES]
    elif audio.shape[0] < TARGET_SAMPLES:
        audio = np.pad(audio, (0, TARGET_SAMPLES - audio.shape[0]))

    # 3) TF log-mel extraction (identical to training)
    audio_tf = tf.convert_to_tensor(audio, dtype=tf.float32)
    mel = extract_logmel_tf(audio_tf)              # (T, 64, 1)
    mel_np = mel.numpy().astype(np.float32)

    # 4) Add batch dimension for inference
    mel_np = np.expand_dims(mel_np, axis=0)        # (1, T, 64, 1)
    return mel_np

def prediction_loop():
    global latest_probability

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_idx = input_details[0]["index"]
    out_idx = output_details[0]["index"]

    in_dtype = input_details[0]["dtype"]
    out_dtype = output_details[0]["dtype"]

    last_run = 0.0

    while not stop_event.is_set():
        try:
            now = time.time()
            if now - last_run < SLIDING_SPEED:
                time.sleep(0.005)
                continue
            last_run = now

            # Snapshot latest 2s mono window (int16)
            with pred_lock:
                window = pred_buffer.copy()

            # === PREPROCESS===
            x = preprocess_audio_training(window)  # float32 (1,T,64,1)

            # === SET INPUT (handle quantized model too) ===
            if np.issubdtype(in_dtype, np.integer):
                scale, zero_point = input_details[0]["quantization"]
                if scale == 0:
                    raise ValueError("Input scale is 0; invalid quantization params.")
                x_q = np.round(x / scale + zero_point).astype(in_dtype)
                interpreter.set_tensor(in_idx, x_q)
            else:
                interpreter.set_tensor(in_idx, x.astype(in_dtype))

            # === INFERENCE ===
            interpreter.invoke()
            y = interpreter.get_tensor(out_idx)

            # === DEQUANTIZE OUTPUT IF NEEDED ===
            if np.issubdtype(out_dtype, np.integer):
                scale, zero_point = output_details[0]["quantization"]
                y = (y.astype(np.float32) - zero_point) * scale

            prediction = float(np.squeeze(y))  # supports (1,1) or (1,)
            label = 1 if prediction > 0.5 else 0
            latest_probability = prediction

            timestamp = datetime.now(athens_tz).strftime("%H:%M:%S")

            confidence_history.append(latest_probability)
            if len(confidence_history) > CONF_HISTORY_LEN:
                confidence_history.pop(0)

            with angle_lock:
                current_angle = latest_angle

            print(f"[Node {node_id}] Prediction: {label} | Probability: {prediction:.3f} | AoA: {current_angle}° | Timestamp: {timestamp}")

            if label == 1:
                message = json.dumps({
                    "id": node_id,
                    "l": label,
                    "p": round(float(latest_probability), 3),
                    "a": current_angle,
                    "ts": timestamp,
                    "s": None
                })
                lora.send_message(message)

        except Exception as e:
            print(f"\033[91m[PREDICTION LOOP ERROR] {e}\033[0m")
            time.sleep(0.2)


############################################################################################################
################################            AoA GCC-PHAT SECTION            ################################
############################################################################################################
def next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()

def gcc_phat_tau(sig: np.ndarray, ref: np.ndarray, fs: int, max_tau: float, interp: int = 16):
    """
    Estimate time delay tau (seconds) between sig and ref using GCC-PHAT.
    +tau => sig arrives LATER than ref.
    Returns: (tau_sec, peak, peak_ratio)
    """
    sig = sig.astype(np.float32)
    ref = ref.astype(np.float32)

    sig = sig - np.mean(sig)
    ref = ref - np.mean(ref)

    w = np.hanning(len(sig)).astype(np.float32)
    sig = sig * w
    ref = ref * w

    n = len(sig) + len(ref)
    nfft = next_pow2(n) * interp

    SIG = np.fft.rfft(sig, n=nfft)
    REF = np.fft.rfft(ref, n=nfft)

    R = SIG * np.conj(REF)
    R /= (np.abs(R) + 1e-12)

    cc = np.fft.irfft(R, n=nfft)

    max_shift = nfft // 2
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))

    max_shift_allowed = min(int(round(max_tau * fs * interp)), max_shift)
    mid = len(cc) // 2
    cc_win = cc[mid - max_shift_allowed : mid + max_shift_allowed + 1]

    k = int(np.argmax(cc_win))
    peak = float(cc_win[k])

    cc_abs = np.abs(cc_win)
    guard = 3
    lo = max(0, k - guard)
    hi = min(len(cc_abs), k + guard + 1)
    cc_abs2 = cc_abs.copy()
    cc_abs2[lo:hi] = 0.0
    second = float(np.max(cc_abs2) + 1e-12)
    peak_ratio = float((abs(peak) + 1e-12) / second)

    shift = k - max_shift_allowed
    tau = shift / float(fs * interp)

    return tau, peak, peak_ratio


def get_latest_aoa_frame(frame_len: int) -> np.ndarray:
    """
    Returns the latest [frame_len,4] samples from aoa_buffer in chronological order.
    """
    with aoa_lock:
        idx = aoa_write_idx
        if frame_len > AOA_RING_LEN:
            frame_len = AOA_RING_LEN

        start = (idx - frame_len) % AOA_RING_LEN
        if start < idx:
            frame = aoa_buffer[start:idx, :].copy()
        else:
            frame = np.vstack((aoa_buffer[start:, :], aoa_buffer[:idx, :])).copy()

    return frame

def aoa_loop():
    """
    AoA estimation using GCC-PHAT. This is a practical baseline:
      - estimates tau on two orthogonal-ish pairs: (mic1,mic3) and (mic2,mic4)
      - converts to an angle proxy via atan2
    You may refine mapping once you confirm your mic geometry/channel mapping.
    """
    global latest_angle

    max_tau = (MIC_RADIUS_M / C_SOUND)  # conservative bound
    angle_hist = []

    while not stop_event.is_set():
        try:
            frame = get_latest_aoa_frame(AOA_FRAME)  # (N,4), int16
            m1 = frame[:, 0]
            m2 = frame[:, 1]
            m3 = frame[:, 2]
            m4 = frame[:, 3]

            tau13, _, pr13 = gcc_phat_tau(m1, m3, SAMPLE_RATE, max_tau, interp=PHAT_INTERP)
            tau24, _, pr24 = gcc_phat_tau(m2, m4, SAMPLE_RATE, max_tau, interp=PHAT_INTERP)

            if pr13 < AOA_MIN_PEAK_RATIO and pr24 < AOA_MIN_PEAK_RATIO:
                time.sleep(AOA_UPDATE_SEC)
                continue

            # Normalize taus to [-1,1]
            x = np.clip(tau13 / max_tau, -1.0, 1.0)
            y = np.clip(tau24 / max_tau, -1.0, 1.0)

            ang = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0

            angle_hist.append(ang)
            if len(angle_hist) > AOA_SMOOTH_N:
                angle_hist.pop(0)

            smoothed = float(np.median(angle_hist))
            with angle_lock:
                latest_angle = int(round(smoothed))

            time.sleep(AOA_UPDATE_SEC)

        except Exception as e:
            print(f"\033[91m[AOA LOOP ERROR] {e}\033[0m")
            time.sleep(0.2)


############################################################################################################
################################               HEARTBEAT REPORT                ################################
############################################################################################################
def heartbeat_report_loop():
    while not stop_event.is_set():
        temp_value = None
        try:
            temp_output = os.popen("vcgencmd measure_temp").read()
            temp_value = float(temp_output.replace("temp=", "").replace("'C\n", ""))
        except Exception:
            temp_value = None

        cpu_usage = psutil.cpu_percent(interval=1)
        ram_usage = psutil.virtual_memory().percent

        message = json.dumps({
            "id": node_id,
            "sts": 1,
            "tmp": temp_value,
            "cpu": cpu_usage,
            "ram": ram_usage
        })
        try:
            lora.send_message(message)
        except Exception as e:
            print(f"[Heartbeat ERROR] {e}")

        # sleep remaining (cpu_percent already waited ~1s)
        for _ in range(9 * 10):
            if stop_event.is_set():
                break
            time.sleep(0.1)

############################################################################################################
################################                    MAIN                    ################################
############################################################################################################
def shutdown():
    stop_event.set()
    time.sleep(0.2)
    try:
        stream.stop_stream()
        stream.close()
    except Exception:
        pass
    try:
        p.terminate()
    except Exception:
        pass

if __name__ == "__main__":
    time.sleep(1)

    # Start threads
    threading.Thread(target=audio_capture_loop, daemon=True).start()
    threading.Thread(target=aoa_loop, daemon=True).start()
    threading.Thread(target=prediction_loop, daemon=True).start()
    threading.Thread(target=heartbeat_report_loop, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\033[93m[Main] Ctrl+C. Shutting down...\033[0m")
    finally:
        shutdown()

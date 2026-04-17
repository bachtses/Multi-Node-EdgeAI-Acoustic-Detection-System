import os
import time
import json
import threading
import psutil
import numpy as np
import tensorflow as tf
import pyaudio
import tflite_runtime.interpreter as tflite
import requests
from lora import LoRaReceiver
from zoneinfo import ZoneInfo
from datetime import datetime
athens_tz = ZoneInfo("Europe/Athens")
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


############################################################################################################
################################                  CONSTANTS                 ################################
############################################################################################################
NODES = {
    1: {"name": 1},
    2: {"name": 2},
    3: {"name": 3},
}

this_node_id = 0                  # local node id
stop_event = threading.Event()    # used by loops

##### Lora params #####
last_detections = {}

##### Audio params #####
RAW_CHANNELS = 6
SAMPLE_RATE = 16000
AUDIO_LENGTH = 2
TARGET_SAMPLES = SAMPLE_RATE * AUDIO_LENGTH  # 32000

CHUNK = 1024
SLIDING_SPEED = 1.0
CONF_HISTORY_LEN = 50

##### Mel params #####
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64

##### AoA params #####
C_SOUND = 343.0
MIC_RADIUS_M = 0.0271
PHAT_INTERP = 16
AOA_FRAME = 2048
AOA_UPDATE_SEC = 0.05
AOA_SMOOTH_N = 7
AOA_MIN_PEAK_RATIO = 1.2
AOA_RING_LEN = max(AOA_FRAME * 8, 16384)
aoa_buffer = np.zeros((AOA_RING_LEN, 4), dtype=np.int16)
aoa_write_idx = 0
aoa_lock = threading.Lock()

##### Buffers #####
pred_buffer = np.zeros(TARGET_SAMPLES, dtype=np.int16)
pred_lock = threading.Lock()

latest_probability = 0.0
confidence_history = []

latest_angle = -1
angle_lock = threading.Lock()


####################################################################################
####################               STATUS REPORTER               ###################
####################################################################################
status_lock = threading.Lock()
node_temperatures = {}  
node_cpu = {}
node_ram = {}
HEARTBEAT_PERIOD = 10.0         
MISSED_HEARTBEAT_LIMIT = 3      
missed_heartbeats = {nid: 0 for nid in NODES.keys()}
node_last_seen = {nid: 0.0 for nid in NODES.keys()}   
node_last_seen_wall = {nid: "-" for nid in NODES.keys()}  
node_is_up = {nid: False for nid in NODES.keys()}    

def status_reporter():
    interval = 10.0
    next_print = time.monotonic() + interval
    while not stop_event.is_set():
        detection_timeout_timer()
        now_mono = time.monotonic()
        if now_mono >= next_print:
            next_print += interval
            current_time = time.strftime("%H:%M:%S", time.localtime())
            local_cpu = psutil.cpu_percent(interval=None)
            local_ram = psutil.virtual_memory().percent
            try:
                with open("/sys/class/thermal/thermal_zone0/temp","r") as f:
                    local_temp = float(f.read().strip()) / 1000.0
            except:
                local_temp = "-"
            print("┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓")
            print("┃  NodeID  ┃     Status     ┃  Last Seen  ┃  Temp (°C)  ┃ CPU (%) ┃ RAM (%) ┃")
            print("┠━━━━━━━━━━╋━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━╋━━━━━━━━━━━━━╋━━━━━━━━━╋━━━━━━━━━┨")
            print(f"┃ {this_node_id:<8} ┃ \033[92mONLINE\033[0m         ┃ {current_time:<11} ┃ {local_temp:<11.1f} ┃ {local_cpu:<7.1f} ┃ {local_ram:<7.1f} ┃")
            with status_lock:
                for nid  in sorted(NODES.keys()):
                    last_seen_mono = node_last_seen.get(nid , 0.0)
                    if last_seen_mono == 0.0:
                        status = "\033[39mOFFLINE\033[0m"
                        last_seen_str = "-"
                        if node_is_up[nid ]:
                            node_is_up[nid ] = False
                    else:
                        elapsed = now_mono - last_seen_mono
                        expected_missed = int(elapsed // HEARTBEAT_PERIOD)
                        missed_heartbeats[nid ] = min(expected_missed, MISSED_HEARTBEAT_LIMIT + 1)
                        is_down = (missed_heartbeats[nid ] >= MISSED_HEARTBEAT_LIMIT)
                        status = "\033[39mOFFLINE\033[0m" if is_down else "\033[92mONLINE\033[0m"
                        last_seen_str = node_last_seen_wall.get(nid , "-")
                        if is_down and node_is_up[nid ]:
                            node_is_up[nid ] = False
                        elif (not is_down) and (not node_is_up[nid ]):
                            node_is_up[nid ] = True
                    temp = node_temperatures.get(nid , "-")
                    cpu = node_cpu.get(nid , "-")
                    ram = node_ram.get(nid , "-")
                    print(f"┃ {nid:<8} ┃ {status:<23} ┃ {last_seen_str:<11} ┃ {temp:<11} ┃ {cpu:<7} ┃ {ram:<7} ┃")
            print("┗━━━━━━━━━━┻━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━┻━━━━━━━━━━━━━┻━━━━━━━━━┻━━━━━━━━━┛")
        time.sleep(0.2)


####################################################################################
####################              AGGREGATOR SERVER             ####################
####################################################################################
def start_aggregator_server():
    print("\033[96m[Aggregator] Ready to receive LoRa messages...\033[0m")
    lora = LoRaReceiver()
    lora.setup_module()
    for raw_data in lora.listen(stop_event=stop_event):
        if stop_event.is_set():
            break
        #print("RECEIVED: ", raw_data)
        try:
            # normalize to string
            if isinstance(raw_data, bytes):
                data = raw_data.decode("utf-8", errors="ignore")
            else:
                data = str(raw_data)
            # remove ASCII control chars except whitespace/newlines (optional)
            data = "".join(ch for ch in data if ch >= " " or ch in "\n\r\t").strip()
            # skip non-json lines (RSSI/SNR/etc)
            if not data.startswith("{") or not data.endswith("}"):
                continue
            # parse one JSON per line
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                continue

            # === process message ===
            node_id = int(msg.get("id", -1))

            if "sts" in msg:  # heartbeat
                with status_lock:
                    node_last_seen[node_id] = time.monotonic()
                    node_last_seen_wall[node_id] = time.strftime("%H:%M:%S", time.localtime())
                    
                temp = msg.get("tmp")
                if temp is not None:
                    node_temperatures[node_id] = temp
                    if temp >= 75:
                        print(f"\033[91m[Aggregator] ⚠️ Node {node_id} overheating: {temp}°C\033[0m")

                cpu = msg.get("cpu")
                if cpu is not None:
                    node_cpu[node_id] = cpu
                    if cpu >= 85:
                        print(f"\033[91m[Aggregator] ⚠️ Node {node_id} high CPU usage: {cpu}%\033[0m")

                ram = msg.get("ram")
                if ram is not None:
                    node_ram[node_id] = ram
                    if ram >= 90:
                        print(f"\033[91m[Aggregator] ⚠️ Node {node_id} high RAM usage: {ram}%\033[0m")
                continue

            label = msg.get("l")
            prob = msg.get("p")
            angle = msg.get("a", -1)
            spectrogram = msg.get("s", None)
            remote_timestamp = msg.get("ts")

            node_info = NODES.get(node_id, {"name": f"Node {node_id}", "location": (0, 0)})
            node_name = node_info["name"]

            try:
                label_i = int(label)
            except:
                label_i = 0

            try:
                prob_f = float(prob)
            except:
                prob_f = 0.0

            try:
                angle_i = int(angle)
            except:
                angle_i = -1

            update_json_status(node_id, label_i == 1, prob_f, angle_i)

            # print only when prediction == 1 for remote nodes
            if label_i == 1:
                print(f"[Node {node_id}] Prediction: {label_i} | Probability: {prob_f:.3f} | AoA: {angle_i:>3}° | Timestamp: {remote_timestamp}")

            with status_lock:
                node_last_seen[node_id] = time.monotonic()
                node_last_seen_wall[node_id] = time.strftime("%H:%M:%S", time.localtime())

            if label_i == 1:
                last_detections[node_name] = {
                    "prob": prob_f,
                    "timestamp": time.time(),
                    "spectrogram": spectrogram,
                    "angle": angle_i
                }

        except Exception as e:
            print(f"\033[91m[Aggregator] Failed to process LoRa message: {e}\033[0m")



############################################################################################################
################################                LOAD MODEL                  ################################
############################################################################################################
try:
    interpreter = tflite.Interpreter(
        # model_path="model/model_ours_cnn2D_logmel_20251219_1629.tflite" alternative model
        model_path="model/model_ours_cnn2D_logmel_20260309_1149_v11.tflite"
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
    print("\033[91m[Microphone] Error: ReSpeaker microphone not found.\033[0m")
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
print("\033[96m[Microphone] Audio stream opened (6 channels, int16).\033[0m")



############################################################################################################
################################             SOUND CAPTURE THREAD           ################################
############################################################################################################
def audio_capture_loop():
    """
    Only thread that calls stream.read().
    Updates:
      - pred_buffer: mono merged sliding 2s window
      - aoa_buffer: ring buffer of 4 mics
    """
    global aoa_write_idx

    while not stop_event.is_set():
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            multi = np.frombuffer(data, dtype=np.int16).astype(np.int32)

            mic1 = multi[1::RAW_CHANNELS]
            mic2 = multi[2::RAW_CHANNELS]
            mic3 = multi[3::RAW_CHANNELS]
            mic4 = multi[4::RAW_CHANNELS]

            mics_4 = np.stack(
                [
                    mic1.astype(np.int16),
                    mic2.astype(np.int16),
                    mic3.astype(np.int16),
                    mic4.astype(np.int16),
                ],
                axis=1,
            )  # (CHUNK, 4)

            merged = (mic1 + mic2 + mic3 + mic4) // 4
            merged = np.clip(merged, -32768, 32767).astype(np.int16)

            # pred buffer (IN-PLACE shift)
            with pred_lock:
                shift = merged.shape[0]
                if shift >= TARGET_SAMPLES:
                    pred_buffer[:] = merged[-TARGET_SAMPLES:]
                else:
                    pred_buffer[:-shift] = pred_buffer[shift:]
                    pred_buffer[-shift:] = merged

            # aoa ring buffer
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
            if stop_event.is_set():
                break
            print(f"\033[91m[CAPTURE LOOP ERROR] {e}\033[0m")
            time.sleep(0.05)





############################################################################################################
################################           AI PREDICTIONS SECTION           ################################
############################################################################################################
def extract_logmel_tf(audio_1d_float32: tf.Tensor) -> tf.Tensor:
    stft = tf.signal.stft(
        audio_1d_float32,
        frame_length=N_FFT,
        frame_step=HOP_LENGTH,
        fft_length=N_FFT,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )
    magnitude = tf.abs(stft)

    mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=N_FFT // 2 + 1,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=0.0,
        upper_edge_hertz=SAMPLE_RATE / 2,
    )

    mel_spec = tf.matmul(magnitude, mel_filterbank)
    mel_spec = tf.math.log(mel_spec + 1e-6)
    mel_spec = tf.expand_dims(mel_spec, axis=-1)
    return mel_spec

def preprocess_audio_training(audio_int16_or_float: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio_int16_or_float)

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)

    if audio.shape[0] > TARGET_SAMPLES:
        audio = audio[:TARGET_SAMPLES]
    elif audio.shape[0] < TARGET_SAMPLES:
        audio = np.pad(audio, (0, TARGET_SAMPLES - audio.shape[0]))

    mel = extract_logmel_tf(tf.convert_to_tensor(audio, dtype=tf.float32))
    mel_np = mel.numpy().astype(np.float32)
    mel_np = np.expand_dims(mel_np, axis=0)
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

            with pred_lock:
                window = pred_buffer.copy()

            x = preprocess_audio_training(window)

            if np.issubdtype(in_dtype, np.integer):
                scale, zero_point = input_details[0]["quantization"]
                if scale == 0:
                    raise ValueError("Input scale is 0; invalid quantization params.")
                x_q = np.round(x / scale + zero_point).astype(in_dtype)
                interpreter.set_tensor(in_idx, x_q)
            else:
                interpreter.set_tensor(in_idx, x.astype(in_dtype))

            interpreter.invoke()
            y = interpreter.get_tensor(out_idx)

            if np.issubdtype(out_dtype, np.integer):
                scale, zero_point = output_details[0]["quantization"]
                y = (y.astype(np.float32) - zero_point) * scale

            prediction = float(np.squeeze(y))
            label = 1 if prediction > 0.5 else 0
            latest_probability = prediction

            this_node_timestamp = datetime.now(athens_tz).strftime("%H:%M:%S")

            confidence_history.append(latest_probability)
            if len(confidence_history) > CONF_HISTORY_LEN:
                confidence_history.pop(0)

            with angle_lock:
                current_angle = latest_angle
            
            update_json_status(this_node_id, label == 1, latest_probability, current_angle)

            # print only when prediction == 1 for node 0
            if label == 1:
                print(f"[Node {this_node_id}] Prediction: {label} | Probability: {prediction:.3f} | AoA: {current_angle:>3}° | Timestamp: {this_node_timestamp}") 


        except Exception as e:
            print(f"\033[91m[PREDICTION LOOP ERROR] {e}\033[0m")
            time.sleep(0.2)


############################################################################################################
################################            AoA GCC-PHAT SECTION            ################################
############################################################################################################
def next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()

def gcc_phat_tau(sig: np.ndarray, ref: np.ndarray, fs: int, max_tau: float, interp: int = 16):
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
    global latest_angle

    max_tau = (MIC_RADIUS_M / C_SOUND)
    angle_hist = []

    while not stop_event.is_set():
        try:
            frame = get_latest_aoa_frame(AOA_FRAME)
            m1, m2, m3, m4 = frame[:, 0], frame[:, 1], frame[:, 2], frame[:, 3]

            tau13, _, pr13 = gcc_phat_tau(m1, m3, SAMPLE_RATE, max_tau, interp=PHAT_INTERP)
            tau24, _, pr24 = gcc_phat_tau(m2, m4, SAMPLE_RATE, max_tau, interp=PHAT_INTERP)

            if pr13 < AOA_MIN_PEAK_RATIO and pr24 < AOA_MIN_PEAK_RATIO:
                time.sleep(AOA_UPDATE_SEC)
                continue

            x = np.clip(tau13 / max_tau, -1.0, 1.0)
            y = np.clip(tau24 / max_tau, -1.0, 1.0)
            ang = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0

            angle_hist.append(ang)
            if len(angle_hist) > AOA_SMOOTH_N:
                angle_hist.pop(0)

            with angle_lock:
                latest_angle = int(round(float(np.median(angle_hist))))

            time.sleep(AOA_UPDATE_SEC)

        except Exception as e:
            print(f"\033[91m[AOA LOOP ERROR] {e}\033[0m")
            time.sleep(0.2)



############################################################################################################
################################            SEND JSON TO PLATFORM            ###############################
############################################################################################################
DETECTION_END_THRESHOLD = 3
JSON_ENDPOINT = "https://serverexample/api/import"
JSON_PASSWORD = "xxxxxxxxxxxxxxx"

def send_json(node, probability, angle):
    payload = {
        "password": JSON_PASSWORD,
        "station_id": 1,
        "sensor_id": 6,
        "type": "acoustic_fusion",
        "data": [
            {
                "objectId": 1,
                "detected": 1,
                "node": node,
                "confidence": float(f"{probability:.2f}"),
                "angle": int(angle),
                "unknown": True,
            }
        ]
    }
    # Print the JSON before sending
    #print("\033[93m[DEBUG] JSON Payload Preview:\033[0m")
    #print(json.dumps(payload, indent=2))
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(JSON_ENDPOINT, json=payload, headers=headers, timeout=10)
        print(f"\033[103m[Aggregator] Drone Detection: Start | Node {node} | Json Sent | Status: {response.status_code}       \033[0m")
        
    except Exception as e:
        print(f"\033[91m[Json Sent] Error sending detection: {e}\033[0m")


# end of detection
def send_end_json(node, probability, angle):
    payload = {
        "password": JSON_PASSWORD,
        "station_id": 1,
        "sensor_id": 7,
        "type": "acoustic_fusion",
        "data": [
            {
                "objectId": 1,
                "detected": 0,
                "node": node,
                "confidence": float(f"{probability:.2f}"),
                "angle": int(angle),
                "unknown": True,
            }
        ]
    }
    #print("\033[93m[DEBUG] End JSON Payload Preview:\033[0m")
    #print(json.dumps(payload, indent=2))
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(JSON_ENDPOINT, json=payload, headers=headers, timeout=10)
        print(f"\033[100m[Aggregator] Drone Detection: End   | Node {node} | Json Sent | Status: {response.status_code}       \033[0m")
    except Exception as e:
        print(f"\033[91m[Json Sent] Error sending end-of-detection: {e}\033[0m")


# buffers
DETECTION_START_REQUIRED = 2
DETECTION_START_WINDOW_SEC = 2.0
DETECTION_END_GAP_SEC = 3.0
node_detection_status = {}
node_status_lock = threading.Lock()
def update_json_status(node_id, detected, probability, angle):
    """
    Called whenever a node produces a prediction.
    Only detection events (detected=True) update the state logic
    """
    if not detected:
        return

    now = time.monotonic()

    with node_status_lock:
        st = node_detection_status.get(node_id, {
            "active": False,
            "arm_count": 0,
            "arm_t0": 0.0,
            "last_det": 0.0,
            "last_prob": 0.0,
            "last_angle": 0
        })

        # update latest detection info
        st["last_det"] = now
        st["last_prob"] = float(probability) if probability is not None else 0.0
        try:
            st["last_angle"] = int(angle)
        except:
            st["last_angle"] = 0

        # --- START LOGIC ---
        if not st["active"]:

            if st["arm_count"] == 0 or (now - st["arm_t0"]) > DETECTION_START_WINDOW_SEC:
                # start new arming window
                st["arm_t0"] = now
                st["arm_count"] = 1
            else:
                st["arm_count"] += 1

            if st["arm_count"] >= DETECTION_START_REQUIRED:
                st["active"] = True
                st["arm_count"] = 0
                st["arm_t0"] = 0.0

                threading.Thread(
                    target=send_json,
                    args=(node_id, st["last_prob"], st["last_angle"]),
                    daemon=True
                ).start()

        node_detection_status[node_id] = st


def detection_timeout_timer():
    """
    Must be called periodically (for example inside status_reporter()).
    Ends detections when silence exceeds DETECTION_END_GAP_SEC.
    """
    now = time.monotonic()
    to_end = []
    with node_status_lock:
        for node_id, st in node_detection_status.items():
            if st.get("active"):
                if (now - st["last_det"]) >= DETECTION_END_GAP_SEC:
                    st["active"] = False
                    st["arm_count"] = 0
                    st["arm_t0"] = 0.0
                    node_detection_status[node_id] = st
                    to_end.append(
                        (node_id, st["last_prob"], st["last_angle"])
                    )

    for node_id, prob, ang in to_end:
        threading.Thread(
            target=send_end_json,
            args=(node_id, prob, ang),
            daemon=True
        ).start()



############################################################################################################
################################                    MAIN                    ################################
############################################################################################################
if __name__ == "__main__":
    threads = [
        threading.Thread(target=audio_capture_loop, daemon=True, name="audio_capture"),
        threading.Thread(target=aoa_loop, daemon=True, name="aoa"),
        threading.Thread(target=prediction_loop, daemon=True, name="prediction"),
        threading.Thread(target=status_reporter, daemon=True, name="status"),
        threading.Thread(target=start_aggregator_server, daemon=True, name="lora_rx"),
    ]

    for thread in threads:
        thread.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[Main] Shutting down...")
        stop_event.set()
    finally:
        try:
            stream.stop_stream()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass
        try:
            p.terminate()
        except Exception:
            pass
        print("[Main] Shutdown complete.")


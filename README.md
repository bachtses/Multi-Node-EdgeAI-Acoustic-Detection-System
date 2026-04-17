## Multi-Node-EdgeAI-Acoustic-Detection-System

## Overview

This project presents a distributed acoustic surveillance system for real-time drone detection using edge AI and low-bandwidth communication.
The system is designed to operate in environments where visual or radar-based detection is limited, leveraging the unique acoustic signature of UAVs.
It consists of multiple edge nodes performing local inference, a central aggregator for fusion and decision-making, and a scalable architecture suitable for large-area monitoring.

<img width="973" height="461" alt="image" src="https://github.com/user-attachments/assets/b6ed7505-05be-4723-b885-01c1f6b8f5a4" />


## System Architecture

The system follows a distributed architecture:

- **Peripheral Nodes (Edge Devices)**
  - Capture audio using multi-microphone arrays
  - Perform real-time signal processing
  - Run local AI inference (TensorFlow Lite)
  - Estimate direction of arrival (AoA)
  - Transmit detection events via LoRa

- **Aggregator Node**
  - Receives detections from all nodes
  - Applies fusion logic and filtering
  - Tracks node activity and system health
  - Sends structured detection events to external APIs
  - Provides visualization (Flask)
  - Displays spectrograms and detections
  - Acts as monitoring interface


## Key Features

- Real-time acoustic drone detection
- Distributed edge AI inference (low latency)
- LoRa-based long-range communication
- Multi-node detection fusion
- Direction of Arrival (AoA) estimation
- Scalable architecture (plug-and-play nodes)
- System health monitoring (CPU, RAM, temperature)
- Low-cost hardware deployment


## Signal Processing Pipeline

Each node processes audio using:

1. Audio acquisition (PyAudio)
2. Multi-channel processing (ReSpeaker)
3. Feature extraction:
   - STFT
   - Log-Mel Spectrogram
4. AI inference (CNN model)
5. AoA estimation (GCC-PHAT)
6. Event generation (JSON)

<img width="833" height="364" alt="image" src="https://github.com/user-attachments/assets/595fca49-3241-4fd7-b65d-ac1a7f34cad8" />

<img width="984" height="222" alt="image" src="https://github.com/user-attachments/assets/b124f212-8f70-4216-ab76-3ad5b4aab38a" />


## AI Model

- Input: Log-Mel Spectrogram (2 sec audio)
- Architecture: 2D CNN
- Task: Binary classification (Drone / No Drone)
- Deployment: TensorFlow Lite (edge optimized)

Optimized for:
- Real-time inference
- Low computational cost
- Robustness in noisy environments

Model trained on:
- Custom dataset (~16,000 samples)
- Real-world outdoor recordings

<img width="976" height="492" alt="image" src="https://github.com/user-attachments/assets/77b2d576-4752-4ae3-8510-8f3853ab5974" />


## AoA Detection Algorithm

Angle of Arrival (AoA) estimation is used to determine the direction from which a sound source (e.g., a drone) is arriving relative to the microphone array. This provides spatial awareness and helps localize the drone in the monitored area. 

The GCC-PHAT (Generalized Cross-Correlation with Phase Transform) algorithm estimates the time delay between signals captured by multiple microphones. By analyzing these time differences, it computes the direction of arrival of the acoustic signal with robustness to noise.

<img width="824" height="317" alt="image" src="https://github.com/user-attachments/assets/dd0c8607-b9c0-4e5a-9be2-1883ffef1c90" />


## Communication (LoRa)

- Lightweight JSON messages (~40 bytes)
- Includes:
  - Node ID
  - Detection label
  - Confidence score
  - AoA estimation
  - Timestamp

Designed for:
- Low bandwidth environments
- Long-range communication
- Energy efficiency

<img width="795" height="636" alt="image" src="https://github.com/user-attachments/assets/47012160-57bc-4538-bef5-ba5d1ec1d8d5" />


## Hardware

Each node includes:

- Raspberry Pi 5
- ReSpeaker 4-Mic Array
- LoRa communication module
- Power supply (battery or wired)

Optimized for outdoor deployment

<img width="972" height="501" alt="image" src="https://github.com/user-attachments/assets/2e866772-f1f2-4f97-8646-312c6dafab17" />

<img width="279" height="439" alt="image" src="https://github.com/user-attachments/assets/8eeb4d12-8f55-4e1b-894c-108350385dcf" />


## Software Components

- `peripheral_node.py` → edge detection logic
- `aggregator.py` → fusion + communication
- `model.tflite` → deployed AI model
- `lora.py` → communication module
- `tuning.py` → configuration

<img width="955" height="535" alt="image" src="https://github.com/user-attachments/assets/6dd53127-16d9-46f6-8a07-fbb0dafc5768" />


## System Capabilities

- Multi-node spatial awareness
- Detection validation via redundancy
- Real-time monitoring of node status
- Event-based detection logic (start/end events)
- Robust operation in noisy environments


## Experimental Setup

- 12,000 labeled audio samples
- Balanced dataset (Drone / Background)
- Real-world recordings (wind, traffic, noise)
- Field testing with DJI drones

<img width="738" height="462" alt="image" src="https://github.com/user-attachments/assets/94103a9a-0ee3-4e5c-a4e3-551f08ca9c96" />


## How to Run

Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Run peripheral node
sudo venv/bin/python3 peripheral_node.py

Run aggregator
python3 aggregator.py

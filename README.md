## Multi-Node-EdgeAI-Acoustic-Detection-System

## Overview

This project presents a distributed acoustic surveillance system for real-time drone detection using edge AI and low-bandwidth communication.
The system is designed to operate in environments where visual or radar-based detection is limited, leveraging the unique acoustic signature of UAVs.
It consists of multiple edge nodes performing local inference, a central aggregator for fusion and decision-making, and a scalable architecture suitable for large-area monitoring.


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


## Hardware

Each node includes:

- Raspberry Pi 5
- ReSpeaker 4-Mic Array
- LoRa communication module
- Power supply (battery or wired)

Optimized for outdoor deployment


## Software Components

- `peripheral_node.py` → edge detection logic
- `aggregator.py` → fusion + communication
- `model.tflite` → deployed AI model
- `lora.py` → communication module
- `tuning.py` → configuration


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


## How to Run

Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Run peripheral node
sudo venv/bin/python3 peripheral_node.py

Run aggregator
python3 aggregator.py

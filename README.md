# Real-Time Voice Activity Detection (VAD)

A lightweight Voice Activity Detection system optimized for edge devices. Detects human speech in audio streams in real-time.

## 🎯 What It Does

- Detects speech vs silence in real-time audio
- Energy-based detection (no heavy AI needed)
- Chunk processing for streaming
- Edge deployment ready (CPU-only)

## 🧠 Why It Matters

VAD is used in:
- Bluetooth earbuds (know when you're speaking)
- Video conferencing (auto-mute when silent)
- Voice assistants (wake on command)
- IoT devices (power saving)

## ⚡ Key Features

- Real-time processing (<50ms latency)
- CPU-only (no GPU)
- Lightweight model
- Works with streaming audio

## 📦 Installation

```bash
pip install -r requirements.txt
```

Or:
```bash
pip install numpy librosa soundfile onnxruntime
```

## 🚀 Usage

### Jupyter Notebook (Recommended)
```bash
jupyter notebook notebook_vad.ipynb
```

### Python Script
```bash
python scripts/vad_inference.py --input input_audio/sample.wav
```

### Real-Time Simulation
```bash
python scripts/realtime_vad.py
```

## 📊 Performance

| Metric | Value |
|--------|-------|
| Latency | <50ms |
| Chunk size | 100ms |
| Platform | CPU only |
| Model size | <1 MB |

## 🔁 Pipeline

```
Audio Stream → Chunking → Energy Calculation → Threshold → Speech/Silence
```

## 📁 Project Structure

```
real-time-voice-activity-detector/
├── notebook_vad.ipynb              # Demo notebook
├── README.md                      # This file
├── requirements.txt              # Dependencies
└── scripts/
    ├── vad_inference.py           # VAD on file
    ├── realtime_vad.py           # Real-time simulation
    └── convert_to_onnx.py      # ONNX conversion
```

## 🎓 Skills Demonstrated

- Audio signal processing
- Real-time systems
- Edge optimization
- Low-latency inference

## 🔗 Related Projects

- [Edge AI Speech Enhancement](../edge-ai-speech-enhancement/)
- [Audio Source Separation](../lightweight-audio-source-separation/)

Together these show a complete audio AI pipeline:

1. **VAD** → Detects speech
2. **Speech Enhancement** → Cleans audio
3. **Source Separation** → Isolates voice

---

*Built for IPHIPI Technologies internship preparation*
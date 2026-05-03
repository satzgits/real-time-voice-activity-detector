# Real-Time Voice Activity Detection (VAD)

Energy-based system for detecting speech in audio streams, optimized for edge devices.

## What It Does

- Detects voice/speech in real-time audio streams
- Chunk-based processing (100ms chunks)
- Optimized for CPU-only inference (<50ms latency)
- Works on IoT and edge devices

## Why It Matters

Voice activity detection is essential for:
- Speech recognition systems
- Voice assistants wake word detection
- Audio recording devices
- Telecommunication systems
- IoT and smart home devices

## Key Features

- Energy-based detection algorithm
- Real-time chunk processing
- ONNX conversion for portable deployment
- CPU-friendly inference
- Low latency (<50ms)

## Installation

```bash
pip install -r requirements.txt
```

Or:
```bash
pip install librosa soundfile numpy
```

## Usage

### Python Script
```bash
python scripts/vad_inference.py --input input_audio/sample.wav --output output_results/vad_results.csv
```

### Real-Time Simulation
```bash
python scripts/realtime_vad.py --chunk 0.1
```

### ONNX Conversion
```bash
python scripts/convert_to_onnx.py
```

## Performance

| Metric | Value |
|--------|-------|
| Chunk Duration | 100ms |
| Inference Latency | <50ms |
| Platform | CPU-only |
| Target Devices | IoT, Edge, Wearables |

## Pipeline

```
Audio Stream → Chunking (100ms) → Energy Calculation → Threshold Check → Speech/No Speech
```

## Project Structure

```
real-time-voice-activity-detector/
├── notebook_vad.ipynb            # Demo notebook
├── README.md                     # This file
├── requirements.txt              # Dependencies
├── LICENSE                       # MIT License
└── scripts/
    ├── vad_inference.py          # Batch VAD
    ├── realtime_vad.py           # Real-time VAD
    └── convert_to_onnx.py        # ONNX conversion
```

## Skills Demonstrated

- Real-time audio processing
- Energy-based signal detection
- Chunk processing for streaming
- Model optimization for edge
- IoT/embedded systems

## Related Projects

- [Edge AI Speech Enhancement](../edge-ai-speech-enhancement/)
- [Audio Source Separation](../lightweight-audio-source-separation/)

Together these show a complete audio AI pipeline!

---


*GitHub: github.com/satzgits/real-time-voice-activity-detector*

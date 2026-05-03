import numpy as np
import librosa
import soundfile as sf
import argparse
from pathlib import Path
import time

def extract_features(audio_chunk, sr=16000):
    """Extract simple features for VAD."""
    energy = np.sum(audio_chunk ** 2)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_chunk))) > 0)
    return energy, zero_crossings

def energy_vad(audio_chunk, sr=16000, threshold=0.01):
    """Energy-based voice activity detection."""
    energy, _ = extract_features(audio_chunk, sr)
    is_speech = energy > threshold
    return is_speech, energy

def process_audio_file(input_path, output_path=None, chunk_duration=1.0, threshold=0.01):
    """Process entire audio file and detect speech segments."""
    print(f"Loading audio from: {input_path}")
    
    audio, sr = librosa.load(input_path, sr=16000)
    chunk_samples = int(chunk_duration * sr)
    
    results = []
    speech_timestamps = []
    
    print(f"Processing in {chunk_duration}s chunks...")
    
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        if len(chunk) < chunk_samples:
            break
            
        is_speech, energy = energy_vad(chunk, sr, threshold)
        results.append({
            'start_time': i / sr,
            'end_time': (i + chunk_samples) / sr,
            'is_speech': is_speech,
            'energy': energy
        })
        
        if is_speech:
            speech_timestamps.append((i / sr, (i + chunk_samples) / sr))
    
    total_duration = len(audio) / sr
    speech_duration = sum(r['end_time'] - r['start_time'] for r in results if r['is_speech'])
    
    print(f"\n=== VAD Results ===")
    print(f"Total duration: {total_duration:.2f}s")
    print(f"Speech detected: {speech_duration:.2f}s ({100*speech_duration/total_duration:.1f}%)")
    print(f"Speech segments: {len(speech_timestamps)}")
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write("start_time,end_time,energy,is_speech\n")
            for r in results:
                f.write(f"{r['start_time']:.3f},{r['end_time']:.3f},{r['energy']:.6f},{r['is_speech']}\n")
        print(f"Results saved to: {output_path}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Voice Activity Detection')
    parser.add_argument('--input', type=str, default='input_audio/sample.wav',
                        help='Input audio file')
    parser.add_argument('--output', type=str, default='output_results/vad_results.csv',
                        help='Output results file')
    parser.add_argument('--chunk', type=float, default=1.0,
                        help='Chunk duration in seconds')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Energy threshold')
    args = parser.parse_args()
    
    Path(args.input).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    results = process_audio_file(args.input, args.output, args.chunk, args.threshold)
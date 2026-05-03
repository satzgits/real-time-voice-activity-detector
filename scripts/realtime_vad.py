"""
Real-Time VAD Simulation
Simulates real-time voice activity detection
"""

import numpy as np
import time

class RealtimeVAD:
    """Simulate real-time VAD processing"""
    
    def __init__(self, threshold=0.01, chunk_ms=100):
        self.threshold = threshold
        self.chunk_ms = chunk_ms
        self.sample_rate = 16000
        self.chunk_samples = int(self.sample_rate * chunk_ms / 1000)
        
    def process_chunk(self, audio_chunk):
        """Process single audio chunk"""
        energy = np.sum(audio_chunk ** 2) / len(audio_chunk)
        is_speech = energy > self.threshold
        return is_speech, energy
    
    def simulate(self, duration_seconds=5):
        """Simulate real-time processing"""
        print(f"=== Real-Time VAD Simulation ===")
        print(f"Duration: {duration_seconds}s")
        print(f"Chunk size: {self.chunk_ms}ms ({self.chunk_samples} samples)")
        print()
        
        total_chunks = int(duration_seconds * 1000 / self.chunk_ms)
        
        for i in range(total_chunks):
            # Generate test chunk
            audio_chunk = np.random.randn(self.chunk_samples) * 0.1
            
            # Randomly add "speech"
            if np.random.random() > 0.7:
                t = np.linspace(0, self.chunk_ms/1000, self.chunk_samples)
                speech = 0.3 * np.sin(2 * np.pi * 300 * t)
                audio_chunk += speech
            
            # Process
            start_time = time.time()
            is_speech, energy = self.process_chunk(audio_chunk)
            elapsed = (time.time() - start_time) * 1000
            
            if is_speech:
                print(f"Chunk {i}: 🔊 SPEECH (energy: {energy:.4f}, time: {elapsed:.2f}ms)")
            
            # Small delay to simulate real-time
            time.sleep(self.chunk_ms / 1000 * 0.1)
        
        print(f"\n✓ Processed {total_chunks} chunks in real-time")

if __name__ == "__main__":
    vad = RealtimeVAD(threshold=0.02)
    vad.simulate(duration_seconds=3)
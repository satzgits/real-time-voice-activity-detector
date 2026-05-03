"""
Real-Time Voice Activity Detection - ONNX Conversion
Optimized for edge deployment
"""

import torch
import torch.nn as nn
import numpy as np
import os

class EnergyVADModel(nn.Module):
    """Energy-based VAD model for ONNX"""
    
    def __init__(self, threshold=0.01):
        super().__init__()
        self.threshold = threshold
        
    def forward(self, audio_chunk):
        """
        Args:
            audio_chunk: Raw audio (batch, samples)
        Returns:
            is_speech: Boolean (batch, 1)
        """
        energy = torch.sum(audio_chunk ** 2, dim=1, keepdim=True) / audio_chunk.shape[1]
        is_speech = (energy > self.threshold).float()
        return is_speech

def convert_vad_to_onnx(model_path="vad_model.onnx"):
    """Convert VAD model to ONNX"""
    
    print("Creating VAD model...")
    model = EnergyVADModel(threshold=0.01)
    model.eval()
    
    # Dummy input
    batch_size = 1
    chunk_samples = 1600  # 100ms at 16kHz
    
    audio_chunk = torch.randn(batch_size, chunk_samples)
    
    print(f"Input shape: {audio_chunk.shape}")
    
    # Export
    torch.onnx.export(
        model,
        audio_chunk,
        model_path,
        input_names=['audio_chunk'],
        output_names=['is_speech'],
        dynamic_axes={'audio_chunk': {1: 'samples'}},
        opset_version=11
    )
    
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✓ Saved: {model_path} ({size_mb:.4f} MB)")
    
    return model_path

def run_onnx_vad(model_path, audio_chunk):
    """Run VAD using ONNX Runtime"""
    try:
        import onnxruntime as ort
        
        session = ort.InferenceSession(model_path)
        output = session.run(None, {'audio_chunk': audio_chunk.numpy()})
        
        return output[0] > 0.5
        
    except ImportError:
        print("Install onnxruntime: pip install onnxruntime")
        return None

if __name__ == "__main__":
    convert_vad_to_onnx()
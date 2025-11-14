# test_comprehensive.py
import torch
import librosa
import soundfile as sf
import numpy as np
import transformers
import whisper
import fastapi
import demucs
import asteroid

print("=== COMPREHENSIVE INSTALLATION TEST ===")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Librosa: {librosa.__version__}")
print(f"SoundFile: {sf.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Whisper: Available")
print(f"FastAPI: {fastapi.__version__}")
print(f"Demucs: {demucs.__version__}")
print(f"Asteroid: {asteroid.__version__}")

# Test core functionality
try:
    # Audio processing
    duration = 2.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Test MFCC extraction
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    print(f"✅ MFCC extraction: {mfcc.shape}")
    
    # Test torch operations
    audio_tensor = torch.tensor(audio).unsqueeze(0)
    print(f"✅ Tensor operations: {audio_tensor.shape}")
    
    # Test audio I/O
    sf.write('test_audio.wav', audio, sample_rate)
    loaded_audio, sr = sf.read('test_audio.wav')
    print(f"✅ Audio I/O: {len(loaded_audio)} samples")
    
    # Test transformers
    from transformers import pipeline
    print("✅ Transformers pipeline available")
    
    print("\n🎉 ALL CORE DEPENDENCIES ARE READY!")
    print("✅ Project requirements are satisfied")
    print("✅ You can proceed with Step 6: Speaker Recognition")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
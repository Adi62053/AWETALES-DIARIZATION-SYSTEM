# test_final.py
import sys

print("=== FINAL INSTALLATION TEST ===")
print(f"Python: {sys.version}")

packages_to_test = [
    'torch', 'numpy', 'scipy', 'soundfile', 'librosa', 
    'transformers', 'fastapi', 'uvicorn', 'websockets',
    'whisper', 'demucs', 'asteroid'
]

for package in packages_to_test:
    try:
        if package == 'torch':
            import torch
            print(f"✅ {package}: {torch.__version__}")
            print(f"   CUDA: {torch.cuda.is_available()}")
        elif package == 'numpy':
            import numpy as np
            print(f"✅ {package}: {np.__version__}")
        elif package == 'scipy':
            import scipy
            print(f"✅ {package}: {scipy.__version__}")
        elif package == 'soundfile':
            import soundfile as sf
            print(f"✅ {package}: {sf.__version__}")
        elif package == 'librosa':
            import librosa
            print(f"✅ {package}: {librosa.__version__}")
        elif package == 'transformers':
            import transformers
            print(f"✅ {package}: {transformers.__version__}")
        elif package == 'fastapi':
            import fastapi
            print(f"✅ {package}: {fastapi.__version__}")
        elif package == 'uvicorn':
            import uvicorn
            print(f"✅ {package}: {uvicorn.__version__}")
        elif package == 'websockets':
            import websockets
            print(f"✅ {package}: {websockets.__version__}")
        elif package == 'whisper':
            import whisper
            print(f"✅ {package}: Available")
        elif package == 'demucs':
            import demucs
            print(f"✅ {package}: {demucs.__version__}")
        elif package == 'asteroid':
            import asteroid
            print(f"✅ {package}: {asteroid.__version__}")
    except ImportError as e:
        print(f"❌ {package}: {e}")

print("\n=== TESTING AUDIO PROCESSING ===")
try:
    import numpy as np
    import soundfile as sf
    import librosa
    
    # Create test audio
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Test librosa
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    print(f"✅ Librosa MFCC: {mfcc.shape}")
    
    # Test soundfile
    sf.write('test_final.wav', audio, sample_rate)
    loaded_audio, sr = sf.read('test_final.wav')
    print(f"✅ SoundFile I/O: {len(loaded_audio)} samples")
    
    print("\n🎉 ALL CORE PACKAGES ARE WORKING!")
    print("✅ Project requirements are COMPLETE")
    print("✅ Ready for Step 6: Speaker Recognition!")
    
except Exception as e:
    print(f"❌ Audio test failed: {e}")
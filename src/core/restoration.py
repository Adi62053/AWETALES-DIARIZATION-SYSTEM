# src/core/restoration.py
"""
Audio Restoration Module using Apollo enhancement
Improves audio quality, reduces noise, and enhances speech clarity for target speaker.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import time

# Import with proper error handling for all dependencies
try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    logging.warning(f"PyTorch import warning: {e}")

try:
    import noisereduce as nr
    NOISE_REDUCE_AVAILABLE = True
except ImportError as e:
    NOISE_REDUCE_AVAILABLE = False
    logging.warning(f"noisereduce import warning: {e}")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError as e:
    LIBROSA_AVAILABLE = False
    logging.warning(f"librosa import warning: {e}")

try:
    import scipy
    import scipy.signal
    SCIPY_AVAILABLE = True
except ImportError as e:
    SCIPY_AVAILABLE = False
    logging.warning(f"scipy import warning: {e}")

from ..utils.config import get_audio_config

class AudioRestorer:
    """
    Audio restoration and enhancement using Apollo model and traditional DSP techniques.
    Improves speech quality, reduces noise, and enhances target speaker clarity.
    """
    
    def __init__(self, config: Optional[Dict] = None, device: str = "cuda"):
        """
        Initialize Audio Restorer.
        
        Args:
            config: Restoration configuration
            device: Computation device
        """
        self.config = config or self._get_default_config()
        self.device = device
        self.model = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self._load_apollo_model()
        
        # DSP parameters
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.enhancement_strength = self.config.get("enhancement_strength", 0.8)
        
        self.logger.info(f"Audio Restorer initialized (Torch: {TORCH_AVAILABLE}, "
                        f"noisereduce: {NOISE_REDUCE_AVAILABLE}, "
                        f"librosa: {LIBROSA_AVAILABLE})")

    def _get_default_config(self) -> Dict:
        """Get default restoration configuration."""
        return {
            "sample_rate": 16000,
            "enhancement_strength": 0.8,
            "noise_reduction": True,
            "speech_enhancement": True,
            "dynamic_range_compression": True,
            "equalization": True,
            "apollo_model_path": "models/apollo/best_model.pth"
        }

    def _load_apollo_model(self):
        """Load Apollo enhancement model with fallbacks."""
        try:
            # Try to load Apollo model
            model_path = Path(self.config.get("apollo_model_path", ""))
            if model_path.exists() and TORCH_AVAILABLE:
                self.model = torch.jit.load(str(model_path))
                if self.device == "cuda" and torch.cuda.is_available():
                    self.model = self.model.cuda()
                self.model.eval()
                self.logger.info("Apollo enhancement model loaded successfully")
            else:
                self.logger.warning("Apollo model not available, using DSP methods")
                self.model = None
        except Exception as e:
            self.logger.warning(f"Failed to load Apollo model: {e}, using DSP methods")
            self.model = None

    def enhance_audio(self, 
                     audio_data: np.ndarray, 
                     sample_rate: int = 16000,
                     is_target_speaker: bool = True) -> np.ndarray:
        """
        Enhance audio quality using Apollo model and DSP techniques.
        
        Args:
            audio_data: Input audio as numpy array
            sample_rate: Audio sample rate
            is_target_speaker: Whether this is target speaker audio
            
        Returns:
            Enhanced audio array
        """
        if audio_data.size == 0:
            return audio_data
        
        start_time = time.time()
        
        # Ensure proper sample rate
        if sample_rate != self.sample_rate:
            audio_data = self._resample_audio(audio_data, sample_rate, self.sample_rate)
        
        # Apply enhancement pipeline
        enhanced_audio = audio_data.copy()
        
        # Step 1: Noise reduction
        if self.config.get("noise_reduction", True):
            enhanced_audio = self._apply_noise_reduction(enhanced_audio)
        
        # Step 2: Apollo model enhancement (if available)
        if self.model is not None and TORCH_AVAILABLE:
            try:
                enhanced_audio = self._apply_apollo_enhancement(enhanced_audio)
            except Exception as e:
                self.logger.warning(f"Apollo enhancement failed: {e}, using DSP methods")
        
        # Step 3: Speech enhancement
        if self.config.get("speech_enhancement", True) and is_target_speaker:
            enhanced_audio = self._enhance_speech(enhanced_audio)
        
        # Step 4: Dynamic range compression
        if self.config.get("dynamic_range_compression", True):
            enhanced_audio = self._apply_compression(enhanced_audio)
        
        # Step 5: Equalization
        if self.config.get("equalization", True):
            enhanced_audio = self._apply_equalization(enhanced_audio)
        
        # Step 6: Normalization
        enhanced_audio = self._normalize_audio(enhanced_audio)
        
        processing_time = time.time() - start_time
        self.logger.debug(f"Audio enhancement completed in {processing_time:.3f}s")
        
        return enhanced_audio

    def _apply_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply noise reduction using noisereduce or fallback methods."""
        if NOISE_REDUCE_AVAILABLE:
            try:
                # Use noisereduce for advanced noise reduction
                reduced_noise = nr.reduce_noise(
                    y=audio_data,
                    sr=self.sample_rate,
                    prop_decrease=self.enhancement_strength,
                    stationary=True
                )
                return reduced_noise
            except Exception as e:
                self.logger.warning(f"noisereduce failed: {e}, using basic noise reduction")
        
        # Fallback: Basic spectral subtraction
        return self._spectral_subtraction(audio_data)

    def _spectral_subtraction(self, audio_data: np.ndarray) -> np.ndarray:
        """Basic spectral subtraction noise reduction."""
        try:
            if LIBROSA_AVAILABLE:
                # Use librosa for STFT processing
                stft = librosa.stft(audio_data, n_fft=2048, hop_length=512)
                magnitude = np.abs(stft)
                phase = np.angle(stft)
                
                # Estimate noise from first few frames
                noise_frames = min(10, magnitude.shape[1])
                noise_mag = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
                
                # Apply spectral subtraction
                enhanced_mag = magnitude - self.enhancement_strength * noise_mag
                enhanced_mag = np.maximum(enhanced_mag, 0.01 * magnitude)  # Avoid negative values
                
                # Reconstruct audio
                enhanced_stft = enhanced_mag * np.exp(1j * phase)
                enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
                
                return enhanced_audio
            else:
                # Very basic high-pass filter as last resort
                return self._high_pass_filter(audio_data)
        except Exception as e:
            self.logger.warning(f"Spectral subtraction failed: {e}")
            return audio_data

    def _high_pass_filter(self, audio_data: np.ndarray, cutoff: float = 80.0) -> np.ndarray:
        """Simple high-pass filter to remove low-frequency noise."""
        try:
            if LIBROSA_AVAILABLE:
                return librosa.effects.preemphasis(audio_data, coef=0.97)
            elif SCIPY_AVAILABLE:
                # Basic IIR high-pass filter using scipy
                nyquist = self.sample_rate / 2
                normal_cutoff = cutoff / nyquist
                b, a = scipy.signal.butter(4, normal_cutoff, btype='high', analog=False)
                return scipy.signal.filtfilt(b, a, audio_data)
            else:
                # Simple DC removal
                return audio_data - np.mean(audio_data)
        except Exception as e:
            self.logger.warning(f"High-pass filter failed: {e}")
            return audio_data

    def _apply_apollo_enhancement(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply Apollo model enhancement."""
        if not TORCH_AVAILABLE or self.model is None:
            return audio_data
        
        try:
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0).unsqueeze(0)
            if self.device == "cuda" and torch.cuda.is_available():
                audio_tensor = audio_tensor.cuda()
            
            # Apply model
            with torch.no_grad():
                enhanced_tensor = self.model(audio_tensor)
            
            # Convert back to numpy
            enhanced_audio = enhanced_tensor.squeeze().cpu().numpy()
            
            return enhanced_audio
            
        except Exception as e:
            self.logger.error(f"Apollo model enhancement failed: {e}")
            return audio_data

    def _enhance_speech(self, audio_data: np.ndarray) -> np.ndarray:
        """Enhance speech characteristics."""
        try:
            if LIBROSA_AVAILABLE:
                # Use spectral shaping to enhance speech frequencies
                stft = librosa.stft(audio_data, n_fft=2048, hop_length=512)
                freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=2048)
                
                # Create speech enhancement mask
                speech_mask = np.ones_like(stft)
                speech_range = (freqs >= 300) & (freqs <= 3400)
                speech_mask[speech_range, :] *= 1.2  # Boost speech frequencies
                
                enhanced_stft = stft * speech_mask
                enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
                
                return enhanced_audio
            else:
                return audio_data
        except Exception as e:
            self.logger.warning(f"Speech enhancement failed: {e}")
            return audio_data

    def _apply_compression(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply dynamic range compression."""
        try:
            # Simple compression: reduce dynamic range
            threshold = 0.1
            ratio = 2.0
            
            compressed = audio_data.copy()
            above_threshold = np.abs(compressed) > threshold
            compressed[above_threshold] = (
                threshold + (compressed[above_threshold] - threshold) / ratio
            )
            
            return compressed
        except Exception as e:
            self.logger.warning(f"Compression failed: {e}")
            return audio_data

    def _apply_equalization(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply frequency equalization."""
        try:
            if LIBROSA_AVAILABLE:
                # Simple EQ: boost mid frequencies for speech clarity
                stft = librosa.stft(audio_data, n_fft=2048, hop_length=512)
                freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=2048)
                
                # Boost mid frequencies (500-2000 Hz)
                mid_boost = np.ones_like(freqs)
                mid_range = (freqs >= 500) & (freqs <= 2000)
                mid_boost[mid_range] = 1.3
                
                # Apply EQ
                eq_stft = stft * mid_boost.reshape(-1, 1)
                eq_audio = librosa.istft(eq_stft, hop_length=512)
                
                return eq_audio
            else:
                return audio_data
        except Exception as e:
            self.logger.warning(f"Equalization failed: {e}")
            return audio_data

    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio to appropriate levels."""
        try:
            # Peak normalization
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                normalized = audio_data / max_val * 0.9  # Leave some headroom
            else:
                normalized = audio_data
            
            return normalized
        except Exception as e:
            self.logger.warning(f"Normalization failed: {e}")
            return audio_data

    def _resample_audio(self, audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio_data
        
        try:
            if LIBROSA_AVAILABLE:
                return librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)
            elif TORCH_AVAILABLE:
                audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)
                resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
                resampled = resampler(audio_tensor)
                return resampled.squeeze().numpy()
            elif SCIPY_AVAILABLE:
                # Use scipy signal resampling
                num_samples = int(len(audio_data) * target_sr / orig_sr)
                return scipy.signal.resample(audio_data, num_samples)
            else:
                # Basic linear interpolation (fallback)
                orig_length = len(audio_data)
                target_length = int(orig_length * target_sr / orig_sr)
                x_orig = np.linspace(0, 1, orig_length)
                x_target = np.linspace(0, 1, target_length)
                return np.interp(x_target, x_orig, audio_data)
        except Exception as e:
            self.logger.error(f"Resampling failed: {e}")
            return audio_data

    def calculate_quality_metrics(self, original_audio: np.ndarray, enhanced_audio: np.ndarray) -> Dict[str, float]:
        """
        Calculate audio quality improvement metrics.
        
        Args:
            original_audio: Original audio data
            enhanced_audio: Enhanced audio data
            
        Returns:
            Dictionary of quality metrics
        """
        try:
            metrics = {}
            
            # Signal-to-Noise Ratio improvement
            snr_original = self._calculate_snr(original_audio)
            snr_enhanced = self._calculate_snr(enhanced_audio)
            metrics["snr_improvement_db"] = snr_enhanced - snr_original
            
            # Peak Signal-to-Noise Ratio
            metrics["psnr_db"] = self._calculate_psnr(original_audio, enhanced_audio)
            
            # Spectral characteristics
            if LIBROSA_AVAILABLE:
                spectral_centroid_orig = librosa.feature.spectral_centroid(y=original_audio, sr=self.sample_rate).mean()
                spectral_centroid_enh = librosa.feature.spectral_centroid(y=enhanced_audio, sr=self.sample_rate).mean()
                metrics["spectral_centroid_change"] = spectral_centroid_enh - spectral_centroid_orig
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Quality metrics calculation failed: {e}")
            return {}

    def _calculate_snr(self, audio_data: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio."""
        try:
            signal_power = np.mean(audio_data ** 2)
            noise_estimate = np.std(audio_data - np.mean(audio_data))
            noise_power = noise_estimate ** 2
            
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
            else:
                snr = 60.0  # Very high SNR if no noise detected
                
            return snr
        except:
            return 0.0

    def _calculate_psnr(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio."""
        try:
            mse = np.mean((original - enhanced) ** 2)
            if mse == 0:
                return float('inf')
            max_val = np.max(np.abs(original))
            psnr = 20 * np.log10(max_val / np.sqrt(mse))
            return psnr
        except:
            return 0.0

# Factory function
def create_audio_restorer(config: Dict = None, device: str = "cuda") -> AudioRestorer:
    """Create audio restorer instance."""
    return AudioRestorer(config, device)

# Test function
def test_restoration_module():
    """Test the audio restoration module."""
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create test audio (sine wave with noise)
        duration = 1.0  # seconds
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        clean_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        noisy_audio = clean_audio + 0.1 * np.random.normal(size=len(clean_audio))
        
        # Initialize restorer
        restorer = create_audio_restorer({"device": "cpu"})
        
        # Enhance audio
        enhanced_audio = restorer.enhance_audio(noisy_audio, sample_rate)
        
        # Calculate metrics
        metrics = restorer.calculate_quality_metrics(noisy_audio, enhanced_audio)
        print(f"Quality metrics: {metrics}")
        
        print("Audio restoration module test completed successfully")
        
    except Exception as e:
        print(f"Restoration test failed: {e}")

if __name__ == "__main__":
    test_restoration_module()
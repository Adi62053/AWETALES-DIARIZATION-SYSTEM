# src/core/output_manager.py
"""
Output Manager for Awetales Diarization System
Generates diarization.json, manages target_speaker.wav, and handles all output formats
with performance metrics and quality validation.
"""

import json
import wave
import numpy as np
from typing import List, Dict, Optional, Union, Any
import logging
from pathlib import Path
import time
from dataclasses import dataclass, asdict
from datetime import datetime

# Import torch with error handling
try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch/torchaudio not available - audio saving will be limited")

from ..utils.config import config_manager, get_system_config
from ..utils.logger import setup_logger

@dataclass
class DiarizationSegment:
    """Represents a single diarization segment with full metadata."""
    speaker: str
    start: float
    end: float
    text: str
    confidence: float
    language: str
    words: List[Dict]
    emotion: Optional[str] = None
    speaker_embedding: Optional[List[float]] = None

@dataclass
class SystemMetrics:
    """Comprehensive system performance metrics."""
    processing_time: float
    real_time_factor: float
    word_error_rate: float
    diarization_error_rate: float
    target_speaker_precision: float
    si_sdr_improvement: float
    average_latency: float
    memory_usage: float
    gpu_utilization: float

class OutputManager:
    """
    Manages all output generation for the diarization system including:
    - diarization.json with structured transcripts
    - target_speaker.wav audio file
    - Real-time streaming outputs
    - Performance metrics and reports
    """
    
    def __init__(self, output_dir: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize Output Manager.
        
        Args:
            output_dir: Base output directory
            config: Output configuration
        """
        self.config = config or self._get_default_config()
        self.system_config = get_system_config()
        
        # Setup output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(self.system_config.get("paths", {}).get("outputs_dir", "./outputs"))
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logger(__name__)
        
        # Performance tracking
        self.processing_start_time = None
        self.metrics = SystemMetrics(
            processing_time=0.0,
            real_time_factor=0.0,
            word_error_rate=0.0,
            diarization_error_rate=0.0,
            target_speaker_precision=0.0,
            si_sdr_improvement=0.0,
            average_latency=0.0,
            memory_usage=0.0,
            gpu_utilization=0.0
        )
        
        self.logger.info(f"Output Manager initialized with output directory: {self.output_dir}")

    def _get_default_config(self) -> Dict:
        """Get default output configuration."""
        return {
            "output_formats": ["json", "txt", "srt", "csv"],
            "audio_format": "wav",
            "audio_sample_rate": 16000,
            "include_word_timestamps": True,
            "include_confidence_scores": True,
            "include_speaker_embeddings": False,
            "max_segment_duration": 30.0,
            "min_confidence_threshold": 0.3
        }

    def start_processing_session(self, session_id: str):
        """Start a new processing session with timing."""
        self.session_id = session_id
        self.processing_start_time = time.time()
        self.session_dir = self.output_dir / session_id
        self.session_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Started processing session: {session_id}")

    def generate_diarization_json(self,
                                segments: List[DiarizationSegment],
                                audio_duration: float,
                                target_speaker_id: str = "Target") -> Dict[str, Any]:
        """
        Generate comprehensive diarization JSON output.
        
        Args:
            segments: List of diarization segments
            audio_duration: Total audio duration in seconds
            target_speaker_id: ID for target speaker
            
        Returns:
            Structured diarization data
        """
        # Filter and validate segments
        valid_segments = self._validate_segments(segments)
        
        # Build diarization data structure
        diarization_data = {
            "metadata": {
                "session_id": getattr(self, 'session_id', 'unknown'),
                "timestamp": datetime.now().isoformat(),
                "audio_duration": audio_duration,
                "total_segments": len(valid_segments),
                "target_speaker": target_speaker_id,
                "system_version": self.system_config.get("system", {}).get("version", "1.0.0")
            },
            "segments": [],
            "speakers": self._extract_speaker_info(valid_segments),
            "summary": self._generate_summary(valid_segments, audio_duration)
        }
        
        # Add segments with proper formatting
        for segment in valid_segments:
            segment_data = asdict(segment)
            
            # Remove None values for cleaner JSON
            segment_data = {k: v for k, v in segment_data.items() if v is not None}
            
            # Remove speaker embeddings unless explicitly requested
            if not self.config.get("include_speaker_embeddings", False):
                segment_data.pop("speaker_embedding", None)
            
            diarization_data["segments"].append(segment_data)
        
        return diarization_data

    def _validate_segments(self, segments: List[DiarizationSegment]) -> List[DiarizationSegment]:
        """Validate and filter diarization segments."""
        valid_segments = []
        
        for segment in segments:
            # Check confidence threshold
            if segment.confidence < self.config.get("min_confidence_threshold", 0.3):
                continue
            
            # Check segment duration
            segment_duration = segment.end - segment.start
            if segment_duration <= 0:
                continue
                
            # Check for overlapping segments (basic validation)
            if not self._has_overlap(segment, valid_segments):
                valid_segments.append(segment)
        
        # Sort by start time
        valid_segments.sort(key=lambda x: x.start)
        
        return valid_segments

    def _has_overlap(self, segment: DiarizationSegment, existing_segments: List[DiarizationSegment]) -> bool:
        """Check if segment overlaps significantly with existing segments."""
        for existing in existing_segments:
            if (segment.start < existing.end and segment.end > existing.start):
                overlap_duration = min(segment.end, existing.end) - max(segment.start, existing.start)
                segment_duration = segment.end - segment.start
                if overlap_duration / segment_duration > 0.5:  # 50% overlap threshold
                    return True
        return False

    def _extract_speaker_info(self, segments: List[DiarizationSegment]) -> Dict[str, Any]:
        """Extract speaker statistics and information."""
        speakers = {}
        
        for segment in segments:
            speaker = segment.speaker
            if speaker not in speakers:
                speakers[speaker] = {
                    "total_segments": 0,
                    "total_duration": 0.0,
                    "total_words": 0,
                    "average_confidence": 0.0,
                    "segments": []
                }
            
            speaker_data = speakers[speaker]
            speaker_data["total_segments"] += 1
            speaker_data["total_duration"] += (segment.end - segment.start)
            speaker_data["total_words"] += len(segment.words)
            speaker_data["average_confidence"] = (
                (speaker_data["average_confidence"] * (speaker_data["total_segments"] - 1) + segment.confidence) 
                / speaker_data["total_segments"]
            )
            speaker_data["segments"].append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })
        
        return speakers

    def _generate_summary(self, segments: List[DiarizationSegment], audio_duration: float) -> Dict[str, Any]:
        """Generate summary statistics for the diarization."""
        if not segments:
            return {}
        
        total_speech_duration = sum(seg.end - seg.start for seg in segments)
        total_words = sum(len(seg.words) for seg in segments)
        avg_confidence = np.mean([seg.confidence for seg in segments])
        
        return {
            "total_speech_duration": total_speech_duration,
            "total_silence_duration": audio_duration - total_speech_duration,
            "speech_ratio": total_speech_duration / audio_duration,
            "total_words": total_words,
            "words_per_minute": (total_words / total_speech_duration * 60) if total_speech_duration > 0 else 0,
            "average_confidence": avg_confidence,
            "segment_count": len(segments)
        }

    def save_target_speaker_audio(self, 
                                audio_data: np.ndarray, 
                                sample_rate: int = 16000,
                                filename: str = "target_speaker.wav") -> Path:
        """
        Save target speaker audio to WAV file.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Audio sample rate
            filename: Output filename
            
        Returns:
            Path to saved audio file
        """
        output_path = self.session_dir / filename
        
        try:
            if TORCH_AVAILABLE:
                # Use torchaudio for saving (preferred method)
                return self._save_audio_torchaudio(audio_data, sample_rate, output_path)
            else:
                # Fallback to scipy/wave
                return self._save_audio_scipy(audio_data, sample_rate, output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save target speaker audio: {e}")
            raise

    def _save_audio_torchaudio(self, audio_data: np.ndarray, sample_rate: int, output_path: Path) -> Path:
        """Save audio using torchaudio."""
        # Ensure audio data is properly formatted
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize audio if needed
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Save using torchaudio
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)  # Add channel dimension
        torchaudio.save(
            str(output_path),
            audio_tensor,
            sample_rate,
            bits_per_sample=16
        )
        
        self.logger.info(f"Target speaker audio saved to: {output_path}")
        return output_path

    def _save_audio_scipy(self, audio_data: np.ndarray, sample_rate: int, output_path: Path) -> Path:
        """Save audio using scipy/wave (fallback method)."""
        try:
            from scipy.io import wavfile
            # Convert to 16-bit PCM
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wavfile.write(str(output_path), sample_rate, audio_int16)
            
            self.logger.info(f"Target speaker audio saved (scipy): {output_path}")
            return output_path
        except ImportError:
            # Final fallback to wave module
            return self._save_audio_wave(audio_data, sample_rate, output_path)

    def _save_audio_wave(self, audio_data: np.ndarray, sample_rate: int, output_path: Path) -> Path:
        """Save audio using Python wave module (basic fallback)."""
        # Convert to 16-bit PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        with wave.open(str(output_path), 'w') as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 2 bytes = 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        self.logger.info(f"Target speaker audio saved (wave): {output_path}")
        return output_path

    def save_diarization_json(self, 
                            diarization_data: Dict[str, Any],
                            filename: str = "diarization.json") -> Path:
        """
        Save diarization data to JSON file.
        
        Args:
            diarization_data: Structured diarization data
            filename: Output filename
            
        Returns:
            Path to saved JSON file
        """
        output_path = self.session_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(diarization_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Diarization JSON saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to save diarization JSON: {e}")
            raise

    def generate_metrics_report(self, 
                              additional_metrics: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance metrics report.
        
        Args:
            additional_metrics: Additional metrics from processing pipeline
            
        Returns:
            Metrics report dictionary
        """
        if self.processing_start_time:
            processing_time = time.time() - self.processing_start_time
            self.metrics.processing_time = processing_time
        
        # Update metrics with additional data
        if additional_metrics:
            for key, value in additional_metrics.items():
                if hasattr(self.metrics, key):
                    setattr(self.metrics, key, value)
        
        metrics_report = {
            "session_id": getattr(self, 'session_id', 'unknown'),
            "timestamp": datetime.now().isoformat(),
            "performance_metrics": asdict(self.metrics),
            "system_info": {
                "version": self.system_config.get("system", {}).get("version", "1.0.0"),
                "environment": self.system_config.get("system", {}).get("environment", "production")
            },
            "quality_metrics": {
                "wer_status": "PASS" if self.metrics.word_error_rate <= 0.15 else "FAIL",
                "der_status": "PASS" if self.metrics.diarization_error_rate <= 0.10 else "FAIL",
                "latency_status": "PASS" if self.metrics.average_latency <= 500 else "FAIL",
                "precision_status": "PASS" if self.metrics.target_speaker_precision >= 0.90 else "FAIL"
            }
        }
        
        return metrics_report

    def save_metrics_report(self, 
                          metrics_report: Dict[str, Any],
                          filename: str = "metrics_report.json") -> Path:
        """
        Save metrics report to JSON file.
        
        Args:
            metrics_report: Metrics data
            filename: Output filename
            
        Returns:
            Path to saved metrics file
        """
        output_path = self.session_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Metrics report saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to save metrics report: {e}")
            raise

    def generate_streaming_output(self,
                                segment: DiarizationSegment,
                                is_final: bool = False,
                                latency_ms: float = 0.0) -> Dict[str, Any]:
        """
        Generate real-time streaming output for WebSocket.
        
        Args:
            segment: Current diarization segment
            is_final: Whether this is final output
            latency_ms: Processing latency in milliseconds
            
        Returns:
            Streaming output dictionary
        """
        return {
            "session_id": getattr(self, 'session_id', 'unknown'),
            "chunk_start": segment.start,
            "chunk_end": segment.end,
            "speaker": segment.speaker,
            "text": segment.text,
            "partial_text": segment.text if not is_final else "",
            "finalized": is_final,
            "confidence": segment.confidence,
            "latency_ms": latency_ms,
            "timestamp": time.time()
        }

    def export_all_formats(self,
                         diarization_data: Dict[str, Any],
                         target_audio: Optional[np.ndarray] = None,
                         sample_rate: int = 16000) -> Dict[str, Path]:
        """
        Export diarization results in all configured formats.
        
        Args:
            diarization_data: Diarization data structure
            target_audio: Target speaker audio data
            sample_rate: Audio sample rate
            
        Returns:
            Dictionary of exported file paths
        """
        exported_files = {}
        
        # Save JSON (always)
        json_path = self.save_diarization_json(diarization_data)
        exported_files["json"] = json_path
        
        # Save audio if provided
        if target_audio is not None:
            audio_path = self.save_target_speaker_audio(target_audio, sample_rate)
            exported_files["audio"] = audio_path
        
        # Export additional formats
        formats = self.config.get("output_formats", ["json"])
        
        if "txt" in formats:
            txt_path = self._export_txt(diarization_data)
            exported_files["txt"] = txt_path
        
        if "srt" in formats:
            srt_path = self._export_srt(diarization_data)
            exported_files["srt"] = srt_path
        
        if "csv" in formats:
            csv_path = self._export_csv(diarization_data)
            exported_files["csv"] = csv_path
        
        # Generate and save metrics report
        metrics_report = self.generate_metrics_report()
        metrics_path = self.save_metrics_report(metrics_report)
        exported_files["metrics"] = metrics_path
        
        self.logger.info(f"Exported {len(exported_files)} file formats for session: {self.session_id}")
        return exported_files

    def _export_txt(self, diarization_data: Dict[str, Any]) -> Path:
        """Export diarization as readable text file."""
        output_path = self.session_dir / "transcript.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Transcript - Session: {diarization_data['metadata']['session_id']}\n")
            f.write(f"Duration: {diarization_data['metadata']['audio_duration']:.2f}s\n")
            f.write("=" * 50 + "\n\n")
            
            for segment in diarization_data["segments"]:
                f.write(f"[{segment['start']:.2f}-{segment['end']:.2f}] {segment['speaker']}: {segment['text']}\n")
                if segment.get('words'):
                    words_text = " ".join([f"{w['word']}({w['confidence']:.2f})" for w in segment['words']])
                    f.write(f"    Words: {words_text}\n")
                f.write("\n")
        
        return output_path

    def _export_srt(self, diarization_data: Dict[str, Any]) -> Path:
        """Export diarization as SRT subtitle file."""
        output_path = self.session_dir / "subtitles.srt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(diarization_data["segments"], 1):
                start_time = self._format_srt_time(segment['start'])
                end_time = self._format_srt_time(segment['end'])
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['speaker']}: {segment['text']}\n\n")
        
        return output_path

    def _export_csv(self, diarization_data: Dict[str, Any]) -> Path:
        """Export diarization as CSV file."""
        output_path = self.session_dir / "diarization.csv"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("speaker,start,end,text,confidence,word_count\n")
            for segment in diarization_data["segments"]:
                word_count = len(segment.get('words', []))
                f.write(f"\"{segment['speaker']}\",{segment['start']:.3f},{segment['end']:.3f},\"{segment['text']}\",{segment['confidence']:.3f},{word_count}\n")
        
        return output_path

    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds to SRT time format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')

# Factory function
def create_output_manager(output_dir: Optional[str] = None, config: Optional[Dict] = None) -> OutputManager:
    """Create output manager instance."""
    return OutputManager(output_dir, config)

# Test function
def test_output_manager():
    """Test the output manager functionality."""
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create output manager
        output_mgr = create_output_manager("./test_output")
        output_mgr.start_processing_session("test_session_001")
        
        # Create test segments
        test_segments = [
            DiarizationSegment(
                speaker="Target",
                start=0.0,
                end=5.0,
                text="Hello, how are you today?",
                confidence=0.95,
                language="en",
                words=[
                    {"word": "Hello", "start": 0.0, "end": 0.8, "confidence": 0.98},
                    {"word": "how", "start": 0.9, "end": 1.2, "confidence": 0.96}
                ]
            )
        ]
        
        # Generate diarization JSON
        diarization_data = output_mgr.generate_diarization_json(test_segments, 10.0)
        print("Diarization JSON generated successfully")
        
        # Test metrics
        metrics = output_mgr.generate_metrics_report({
            "word_error_rate": 0.12,
            "diarization_error_rate": 0.08
        })
        print(f"Metrics report: {metrics['quality_metrics']}")
        
        print("Output Manager test completed successfully")
        
    except Exception as e:
        print(f"Output Manager test failed: {e}")

if __name__ == "__main__":
    test_output_manager()
"""
Voice Separation Module for Awetales Target Speaker Diarization System

This module handles:
- MossFormer2 voice separation for target speaker isolation
- Multi-speaker separation with target identification
- Speaker embedding integration for target matching
- GPU-optimized separation pipeline

Author: Awetales Engineering Team
Date: 2024
Version: 1.0
"""

import os
import logging
import asyncio
import numpy as np
import torch
import torchaudio
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass
from pathlib import Path
import librosa 
import soundfile as sf 
import json
import time
from scipy import signal
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class SeparationConfig:
    """Configuration for voice separation pipeline"""
    sample_rate: int = 16000
    chunk_duration: float = 2.0
    overlap_duration: float = 0.5
    separation_threshold: float = 0.7
    max_speakers: int = 4
    target_similarity_threshold: float = 0.8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model paths
    mossformer2_model_path: str = "models/mossformer2_separation.pth"
    target_embedding_path: str = "models/target_speaker_embedding.pt"

class MossFormer2Separator:
    """MossFormer2 based voice separation for target speaker isolation"""
    
    def __init__(self, config: SeparationConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self.target_embedding = None
        self._load_model()
        self._load_target_embedding()
    
    def _load_model(self):
        """Load MossFormer2 separation model"""
        try:
            logger.info("Loading MossFormer2 voice separation model...")
            
            # Placeholder for actual MossFormer2 implementation
            # In production, this would load the actual model architecture and weights
            self.model = torch.nn.Sequential(
                torch.nn.Conv1d(1, 256, 512, stride=256, padding=256),
                torch.nn.ReLU(),
                torch.nn.TransformerEncoder(
                    torch.nn.TransformerEncoderLayer(
                        d_model=256,
                        nhead=8,
                        dim_feedforward=1024,
                        dropout=0.1
                    ),
                    num_layers=6
                ),
                torch.nn.Conv1d(256, 512, 512, stride=256, padding=256),
                torch.nn.ReLU(),
                torch.nn.Conv1d(512, self.config.max_speakers, 1)
            ).to(self.device)
            
            # Load model weights if available
            if os.path.exists(self.config.mossformer2_model_path):
                try:
                    state_dict = torch.load(self.config.mossformer2_model_path, 
                                          map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    logger.info("MossFormer2 model weights loaded successfully")
                except Exception as e:
                    logger.warning(f"Could not load MossFormer2 weights: {e}")
            
            self.model.eval()
            logger.info("MossFormer2 separation model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to load MossFormer2 model: {e}")
            raise
    
    def _load_target_embedding(self):
        """Load target speaker embedding for identification"""
        try:
            if os.path.exists(self.config.target_embedding_path):
                self.target_embedding = torch.load(self.config.target_embedding_path, 
                                                 map_location=self.device)
                logger.info("Target speaker embedding loaded successfully")
            else:
                logger.warning("No target speaker embedding found. Target identification will be limited.")
                self.target_embedding = None
                
        except Exception as e:
            logger.error(f"Failed to load target embedding: {e}")
            self.target_embedding = None
    
    async def separate_speakers(self, 
                              audio: np.ndarray,
                              target_embedding: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Separate multiple speakers from audio mixture
        
        Args:
            audio: Input audio mixture
            target_embedding: Optional target speaker embedding for identification
            
        Returns:
            Dictionary containing separated speakers and metadata
        """
        try:
            start_time = time.time()
            
            # Use provided target embedding or loaded one
            current_target_embedding = target_embedding or self.target_embedding
            
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Perform separation
            with torch.no_grad():
                separated_sources = self.model(audio_tensor)
                separated_sources = torch.nn.functional.softmax(separated_sources, dim=1)
            
            # Post-process separated sources
            separation_results = await self._post_process_separation(
                separated_sources, audio, current_target_embedding
            )
            
            processing_time = time.time() - start_time
            
            result = {
                "separated_sources": separation_results,
                "num_speakers_detected": len(separation_results),
                "target_speaker_identified": any(s["is_target"] for s in separation_results),
                "processing_time": processing_time,
                "separation_quality": self._calculate_separation_quality(separation_results)
            }
            
            logger.info(f"Speaker separation completed in {processing_time:.3f}s")
            logger.info(f"Detected {result['num_speakers_detected']} speakers")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in speaker separation: {e}")
            raise
    
    async def _post_process_separation(self,
                                     separated_sources: torch.Tensor,
                                     original_audio: np.ndarray,
                                     target_embedding: Optional[torch.Tensor] = None) -> List[Dict[str, Any]]:
        """Post-process separated sources and identify target speaker"""
        try:
            separated_sources_np = separated_sources.squeeze().cpu().numpy()
            
            # Handle single speaker case
            if len(separated_sources_np.shape) == 1:
                separated_sources_np = separated_sources_np.reshape(1, -1)
            
            results = []
            
            for i in range(min(separated_sources_np.shape[0], self.config.max_speakers)):
                source_audio = separated_sources_np[i]
                
                # Apply threshold and normalize
                source_audio = self._apply_separation_mask(source_audio, original_audio)
                
                # Calculate speaker characteristics
                speaker_embedding = await self._extract_speaker_embedding(source_audio)
                energy_level = np.mean(np.abs(source_audio))
                speech_duration = self._calculate_speech_duration(source_audio)
                
                # Identify if this is target speaker
                is_target = False
                similarity_score = 0.0
                
                if target_embedding is not None and speaker_embedding is not None:
                    similarity_score = self._calculate_similarity(
                        speaker_embedding, target_embedding
                    )
                    is_target = similarity_score >= self.config.target_similarity_threshold
                
                speaker_info = {
                    "speaker_id": f"Speaker_{i+1}",
                    "audio": source_audio,
                    "energy_level": energy_level,
                    "speech_duration": speech_duration,
                    "is_target": is_target,
                    "similarity_score": similarity_score,
                    "embedding": speaker_embedding
                }
                
                results.append(speaker_info)
            
            # Sort by energy level (most active speaker first)
            results.sort(key=lambda x: x["energy_level"], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in separation post-processing: {e}")
            return []
    
    def _apply_separation_mask(self, separation_mask: np.ndarray, original_audio: np.ndarray) -> np.ndarray:
        """Apply separation mask to original audio"""
        # Ensure same length
        min_length = min(len(separation_mask), len(original_audio))
        separation_mask = separation_mask[:min_length]
        original_audio = original_audio[:min_length]
        
        # Apply mask with threshold
        mask = separation_mask > self.config.separation_threshold
        separated_audio = original_audio * mask
        
        return separated_audio
    
    async def _extract_speaker_embedding(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract speaker embedding from separated audio"""
        try:
            # Placeholder for actual speaker embedding extraction
            # This would typically use a pre-trained speaker verification model
            
            if len(audio) < 16000:  # Minimum 1 second for reliable embedding
                return None
            
            # Simulate embedding extraction using MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio, 
                sr=self.config.sample_rate, 
                n_mfcc=64, 
                n_fft=512, 
                hop_length=256
            )
            
            # Average over time to get fixed-size embedding
            embedding = np.mean(mfccs, axis=1)
            
            # Normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Could not extract speaker embedding: {e}")
            return None
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            if embedding1 is None or embedding2 is None:
                return 0.0
            
            # Ensure same dimensions
            min_dim = min(embedding1.shape[0], embedding2.shape[0])
            embedding1 = embedding1[:min_dim].reshape(1, -1)
            embedding2 = embedding2[:min_dim].reshape(1, -1)
            
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Error calculating similarity: {e}")
            return 0.0
    
    def _calculate_speech_duration(self, audio: np.ndarray) -> float:
        """Calculate actual speech duration in separated audio"""
        # Simple energy-based speech detection
        energy = np.abs(audio)
        speech_frames = energy > (np.max(energy) * 0.1)  # 10% of max energy
        speech_duration = np.sum(speech_frames) / self.config.sample_rate
        
        return speech_duration
    
    def _calculate_separation_quality(self, separation_results: List[Dict[str, Any]]) -> float:
        """Calculate overall separation quality score"""
        if not separation_results:
            return 0.0
        
        quality_scores = []
        
        for result in separation_results:
            # Score based on energy concentration and speech duration
            energy_score = min(result["energy_level"] * 10, 1.0)  # Normalize
            duration_score = min(result["speech_duration"] / 2.0, 1.0)  # Normalize to 2s max
            
            quality_score = (energy_score + duration_score) / 2
            quality_scores.append(quality_score)
        
        return float(np.mean(quality_scores))
    
    async def isolate_target_speaker(self, 
                                   audio: np.ndarray,
                                   target_embedding: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Specifically isolate target speaker from mixture
        
        Args:
            audio: Input audio mixture
            target_embedding: Target speaker embedding
            
        Returns:
            Target speaker audio and metadata
        """
        try:
            start_time = time.time()
            
            # Perform full separation
            separation_result = await self.separate_speakers(audio, target_embedding)
            
            # Find target speaker
            target_speakers = [s for s in separation_result["separated_sources"] 
                             if s["is_target"]]
            
            if target_speakers:
                target_speaker = target_speakers[0]  # Take the most confident one
            else:
                # Fallback: take speaker with highest energy
                target_speaker = max(separation_result["separated_sources"], 
                                   key=lambda x: x["energy_level"])
                logger.warning("Target speaker not confidently identified, using highest energy speaker")
            
            processing_time = time.time() - start_time
            
            result = {
                "target_audio": target_speaker["audio"],
                "similarity_score": target_speaker["similarity_score"],
                "is_confident_target": target_speaker["is_target"],
                "speaker_id": target_speaker["speaker_id"],
                "processing_time": processing_time,
                "separation_quality": separation_result["separation_quality"],
                "all_speakers_info": separation_result["separated_sources"]
            }
            
            logger.info(f"Target speaker isolation completed in {processing_time:.3f}s")
            logger.info(f"Target confidence: {target_speaker['similarity_score']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in target speaker isolation: {e}")
            raise

class VoiceSeparator:
    """
    Main voice separation class that orchestrates MossFormer2 separation
    and target speaker isolation
    """
    
    def __init__(self, config: SeparationConfig = None):
        self.config = config or SeparationConfig()
        self.separator = MossFormer2Separator(self.config)
        
        logger.info("Voice Separator initialized successfully")
    
    async def process_audio(self, 
                          audio_input: Union[str, np.ndarray, torch.Tensor],
                          sample_rate: int = None,
                          target_embedding: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Main separation pipeline
        
        Args:
            audio_input: Audio file path, numpy array, or torch tensor
            sample_rate: Sample rate of input audio
            target_embedding: Target speaker embedding for identification
            
        Returns:
            Separation results including target speaker audio
        """
        try:
            start_time = time.time()
            
            # Load audio
            audio_array, actual_sr = await self._load_audio(audio_input, sample_rate)
            
            # Resample if necessary
            if actual_sr != self.config.sample_rate:
                audio_array = librosa.resample(audio_array, orig_sr=actual_sr, 
                                             target_sr=self.config.sample_rate)
            
            # Isolate target speaker
            isolation_result = await self.separator.isolate_target_speaker(
                audio_array, target_embedding
            )
            
            # Prepare final output
            total_time = time.time() - start_time
            
            result = {
                "target_speaker_audio": isolation_result["target_audio"],
                "sample_rate": self.config.sample_rate,
                "similarity_score": isolation_result["similarity_score"],
                "is_confident_target": isolation_result["is_confident_target"],
                "speaker_id": isolation_result["speaker_id"],
                "separation_quality": isolation_result["separation_quality"],
                "all_speakers": isolation_result["all_speakers_info"],
                "processing_time": total_time,
                "audio_duration": len(audio_array) / self.config.sample_rate
            }
            
            logger.info(f"Voice separation pipeline completed in {total_time:.3f}s")
            logger.info(f"Target speaker similarity: {result['similarity_score']:.3f}")
            logger.info(f"Separation quality: {result['separation_quality']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in voice separation pipeline: {e}")
            raise
    
    async def _load_audio(self, 
                         audio_input: Union[str, np.ndarray, torch.Tensor],
                         sample_rate: int = None) -> Tuple[np.ndarray, int]:
        """Load audio from various input types"""
        try:
            if isinstance(audio_input, str):
                if not os.path.exists(audio_input):
                    raise FileNotFoundError(f"Audio file not found: {audio_input}")
                
                audio_array, sr = librosa.load(audio_input, sr=None)
                logger.info(f"Loaded audio from {audio_input}, duration: {len(audio_array)/sr:.2f}s")
                
            elif isinstance(audio_input, np.ndarray):
                if sample_rate is None:
                    raise ValueError("sample_rate must be provided for numpy array input")
                audio_array = audio_input
                sr = sample_rate
                
            elif isinstance(audio_input, torch.Tensor):
                if sample_rate is None:
                    raise ValueError("sample_rate must be provided for tensor input")
                audio_array = audio_input.cpu().numpy()
                sr = sample_rate
                
            else:
                raise ValueError(f"Unsupported audio input type: {type(audio_input)}")
            
            # Ensure mono
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=0)
            
            return audio_array, sr
            
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
    
    async def process_streaming_chunk(self, 
                                    audio_chunk: np.ndarray,
                                    target_embedding: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Process a single audio chunk for real-time streaming
        
        Args:
            audio_chunk: Audio chunk for processing
            target_embedding: Target speaker embedding
            
        Returns:
            Separation results for the chunk
        """
        try:
            # Process chunk with MossFormer2
            separation_result = await self.separator.isolate_target_speaker(
                audio_chunk, target_embedding
            )
            
            return {
                "target_audio": separation_result["target_audio"],
                "similarity_score": separation_result["similarity_score"],
                "is_confident_target": separation_result["is_confident_target"],
                "processing_time": separation_result["processing_time"]
            }
            
        except Exception as e:
            logger.error(f"Error processing streaming chunk: {e}")
            # Return original chunk as fallback
            return {
                "target_audio": audio_chunk,
                "similarity_score": 0.0,
                "is_confident_target": False,
                "processing_time": 0.0
            }

# Utility functions
async def create_voice_separator(config: SeparationConfig = None) -> VoiceSeparator:
    """Factory function to create voice separator"""
    return VoiceSeparator(config)

def save_target_speaker_audio(target_audio: np.ndarray, sample_rate: int, output_path: str):
    """Save target speaker audio to file"""
    try:
        sf.write(output_path, target_audio, sample_rate)
        logger.info(f"Target speaker audio saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving target audio to {output_path}: {e}")

def calculate_si_sdr(original: np.ndarray, enhanced: np.ndarray) -> float:
    """Calculate SI-SDR improvement"""
    try:
        # Scale invariant signal-to-distortion ratio
        target = np.dot(enhanced, original) * original / (np.dot(original, original) + 1e-8)
        distortion = enhanced - target
        
        si_sdr = 10 * np.log10(
            np.dot(target, target) / (np.dot(distortion, distortion) + 1e-8)
        )
        return float(si_sdr)
    except Exception as e:
        logger.warning(f"Could not calculate SI-SDR: {e}")
        return 0.0

# Example usage
async def main():
    """Example usage of the voice separator"""
    config = SeparationConfig()
    separator = await create_voice_separator(config)
    
    # Process an audio file
    result = await separator.process_audio("path/to/mixture_audio.wav")
    
    # Save target speaker audio
    save_target_speaker_audio(
        result["target_speaker_audio"],
        result["sample_rate"],
        "target_speaker.wav"
    )
    
    print(f"Separation completed in {result['processing_time']:.2f}s")
    print(f"Target similarity: {result['similarity_score']:.3f}")
    print(f"Separation quality: {result['separation_quality']:.3f}")

if __name__ == "__main__":
    asyncio.run(main())
"""
Speaker Recognition Module for Awetales Target Speaker Diarization System

This module handles:
- ERes2NetV2 speaker embedding extraction
- Target speaker matching and similarity scoring
- Speaker verification and identification
- Embedding database management

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
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import hashlib

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class SpeakerRecogConfig:
    """Configuration for speaker recognition pipeline"""
    sample_rate: int = 16000
    embedding_dim: int = 192  # ERes2NetV2 embedding dimension
    similarity_threshold: float = 0.8
    min_audio_duration: float = 1.0  # Minimum seconds for reliable embedding
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model paths
    eres2net_model_path: str = "models/eres2netv2_speaker.pth"
    
    # Database settings
    embedding_db_path: str = "data/speaker_embeddings.json"
    max_speakers: int = 100

class ERes2NetV2Extractor:
    """ERes2NetV2 based speaker embedding extraction"""
    
    def __init__(self, config: SpeakerRecogConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load ERes2NetV2 model"""
        try:
            logger.info("Loading ERes2NetV2 speaker recognition model...")
            
            # Placeholder for actual ERes2NetV2 implementation
            # In production, this would load the actual model architecture
            self.model = torch.nn.Sequential(
                torch.nn.Conv1d(1, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv1d(64, 128, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool1d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(128, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, self.config.embedding_dim),
                torch.nn.Tanh()  # Normalize embeddings to [-1, 1]
            ).to(self.device)
            
            # Load model weights if available
            if os.path.exists(self.config.eres2net_model_path):
                try:
                    state_dict = torch.load(self.config.eres2net_model_path, 
                                          map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    logger.info("ERes2NetV2 model weights loaded successfully")
                except Exception as e:
                    logger.warning(f"Could not load ERes2NetV2 weights: {e}")
            
            self.model.eval()
            logger.info("ERes2NetV2 model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ERes2NetV2 model: {e}")
            raise
    
    async def extract_embedding(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract speaker embedding from audio
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary containing embedding and metadata
        """
        try:
            start_time = time.time()
            
            # Validate audio length
            if len(audio) < self.config.min_audio_duration * self.config.sample_rate:
                raise ValueError(f"Audio too short: {len(audio)/self.config.sample_rate:.2f}s "
                               f"(min: {self.config.min_audio_duration}s)")
            
            # Preprocess audio
            processed_audio = await self._preprocess_audio(audio)
            
            # Convert to tensor
            audio_tensor = torch.FloatTensor(processed_audio).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model(audio_tensor)
                embedding = embedding.squeeze().cpu().numpy()
            
            # Normalize embedding
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            processing_time = time.time() - start_time
            
            result = {
                "embedding": embedding,
                "embedding_dim": len(embedding),
                "audio_duration": len(audio) / self.config.sample_rate,
                "processing_time": processing_time,
                "embedding_norm": float(np.linalg.norm(embedding))
            }
            
            logger.info(f"Embedding extraction completed in {processing_time:.3f}s")
            logger.info(f"Embedding dimension: {result['embedding_dim']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in embedding extraction: {e}")
            raise
    
    async def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Preprocess audio for embedding extraction"""
        try:
            # Ensure proper length
            target_length = int(self.config.sample_rate * 3.0)  # 3 seconds
            if len(audio) > target_length:
                # Take middle segment
                start = (len(audio) - target_length) // 2
                audio = audio[start:start + target_length]
            elif len(audio) < target_length:
                # Pad with zeros
                pad_length = target_length - len(audio)
                audio = np.pad(audio, (0, pad_length), mode='constant')
            
            # Normalize audio
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            return audio
            
        except Exception as e:
            logger.error(f"Error in audio preprocessing: {e}")
            return audio

class SpeakerDatabase:
    """Speaker embedding database management"""
    
    def __init__(self, config: SpeakerRecogConfig):
        self.config = config
        self.embeddings_db = defaultdict(list)
        self.speaker_info = {}
        self._load_database()
    
    def _load_database(self):
        """Load speaker database from file"""
        try:
            if os.path.exists(self.config.embedding_db_path):
                with open(self.config.embedding_db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.embeddings_db = defaultdict(list, data.get('embeddings', {}))
                    self.speaker_info = data.get('speaker_info', {})
                logger.info(f"Loaded speaker database with {len(self.embeddings_db)} speakers")
            else:
                logger.info("No existing speaker database found, creating new one")
        except Exception as e:
            logger.error(f"Error loading speaker database: {e}")
    
    def save_database(self):
        """Save speaker database to file"""
        try:
            os.makedirs(os.path.dirname(self.config.embedding_db_path), exist_ok=True)
            data = {
                'embeddings': dict(self.embeddings_db),
                'speaker_info': self.speaker_info,
                'timestamp': time.time()
            }
            with open(self.config.embedding_db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Speaker database saved with {len(self.embeddings_db)} speakers")
        except Exception as e:
            logger.error(f"Error saving speaker database: {e}")
    
    def add_speaker(self, speaker_id: str, embedding: np.ndarray, metadata: Dict[str, Any] = None):
        """Add speaker embedding to database"""
        try:
            if speaker_id not in self.embeddings_db:
                self.embeddings_db[speaker_id] = []
            
            # Convert embedding to list for JSON serialization
            embedding_list = embedding.tolist()
            self.embeddings_db[speaker_id].append(embedding_list)
            
            # Store metadata
            if metadata:
                if speaker_id not in self.speaker_info:
                    self.speaker_info[speaker_id] = {}
                self.speaker_info[speaker_id].update(metadata)
            
            logger.info(f"Added embedding for speaker: {speaker_id}")
            
        except Exception as e:
            logger.error(f"Error adding speaker to database: {e}")
    
    def find_similar_speakers(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar speakers in database"""
        try:
            similarities = []
            
            for speaker_id, embeddings in self.embeddings_db.items():
                for i, db_embedding in enumerate(embeddings):
                    db_embedding_np = np.array(db_embedding)
                    similarity = self._calculate_similarity(query_embedding, db_embedding_np)
                    
                    similarities.append({
                        'speaker_id': speaker_id,
                        'similarity': similarity,
                        'embedding_index': i,
                        'is_target': similarity >= self.config.similarity_threshold
                    })
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar speakers: {e}")
            return []
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-8
            )
            return float(similarity)
        except Exception as e:
            logger.warning(f"Error calculating similarity: {e}")
            return 0.0
    
    def get_speaker_count(self) -> int:
        """Get number of speakers in database"""
        return len(self.embeddings_db)

class SpeakerRecognizer:
    """
    Main speaker recognition class that orchestrates embedding extraction
    and speaker matching
    """
    
    def __init__(self, config: SpeakerRecogConfig = None):
        self.config = config or SpeakerRecogConfig()
        self.extractor = ERes2NetV2Extractor(self.config)
        self.database = SpeakerDatabase(self.config)
        
        logger.info("Speaker Recognizer initialized successfully")
    
    async def register_speaker(self, 
                             audio_input: Union[str, np.ndarray, torch.Tensor],
                             speaker_id: str,
                             sample_rate: int = None,
                             metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Register a new speaker in the database
        
        Args:
            audio_input: Audio file path, numpy array, or torch tensor
            speaker_id: Unique identifier for the speaker
            sample_rate: Sample rate of input audio
            metadata: Additional speaker metadata
            
        Returns:
            Registration results
        """
        try:
            start_time = time.time()
            
            # Load and validate audio
            audio_array, actual_sr = await self._load_audio(audio_input, sample_rate)
            
            # Extract embedding
            embedding_result = await self.extractor.extract_embedding(audio_array)
            
            # Add to database
            self.database.add_speaker(speaker_id, embedding_result["embedding"], metadata)
            
            # Save database
            self.database.save_database()
            
            total_time = time.time() - start_time
            
            result = {
                "speaker_id": speaker_id,
                "embedding_dim": embedding_result["embedding_dim"],
                "audio_duration": embedding_result["audio_duration"],
                "processing_time": total_time,
                "database_size": self.database.get_speaker_count(),
                "success": True
            }
            
            logger.info(f"Speaker registration completed in {total_time:.3f}s")
            logger.info(f"Registered speaker: {speaker_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in speaker registration: {e}")
            return {
                "speaker_id": speaker_id,
                "success": False,
                "error": str(e)
            }
    
    async def identify_speaker(self, 
                             audio_input: Union[str, np.ndarray, torch.Tensor],
                             sample_rate: int = None,
                             top_k: int = 3) -> Dict[str, Any]:
        """
        Identify speaker from audio
        
        Args:
            audio_input: Audio file path, numpy array, or torch tensor
            sample_rate: Sample rate of input audio
            top_k: Number of top matches to return
            
        Returns:
            Speaker identification results
        """
        try:
            start_time = time.time()
            
            # Load audio
            audio_array, actual_sr = await self._load_audio(audio_input, sample_rate)
            
            # Extract embedding
            embedding_result = await self.extractor.extract_embedding(audio_array)
            
            # Find similar speakers
            similar_speakers = self.database.find_similar_speakers(
                embedding_result["embedding"], top_k
            )
            
            # Determine best match
            best_match = similar_speakers[0] if similar_speakers else None
            is_identified = best_match and best_match["is_target"] if best_match else False
            
            total_time = time.time() - start_time
            
            result = {
                "best_match": best_match,
                "similar_speakers": similar_speakers,
                "is_identified": is_identified,
                "embedding_dim": embedding_result["embedding_dim"],
                "audio_duration": embedding_result["audio_duration"],
                "processing_time": total_time,
                "database_size": self.database.get_speaker_count()
            }
            
            logger.info(f"Speaker identification completed in {total_time:.3f}s")
            if best_match:
                logger.info(f"Best match: {best_match['speaker_id']} "
                          f"(similarity: {best_match['similarity']:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in speaker identification: {e}")
            return {
                "best_match": None,
                "similar_speakers": [],
                "is_identified": False,
                "error": str(e)
            }
    
    async def verify_speaker(self,
                           audio_input: Union[str, np.ndarray, torch.Tensor],
                           claimed_speaker_id: str,
                           sample_rate: int = None) -> Dict[str, Any]:
        """
        Verify if audio matches claimed speaker
        
        Args:
            audio_input: Audio file path, numpy array, or torch tensor
            claimed_speaker_id: Speaker ID to verify against
            sample_rate: Sample rate of input audio
            
        Returns:
            Verification results
        """
        try:
            start_time = time.time()
            
            # Load audio
            audio_array, actual_sr = await self._load_audio(audio_input, sample_rate)
            
            # Extract embedding
            embedding_result = await self.extractor.extract_embedding(audio_array)
            
            # Check if claimed speaker exists
            if claimed_speaker_id not in self.database.embeddings_db:
                return {
                    "verified": False,
                    "similarity": 0.0,
                    "reason": "Speaker not in database",
                    "processing_time": time.time() - start_time
                }
            
            # Calculate similarity with claimed speaker
            claimed_embeddings = self.database.embeddings_db[claimed_speaker_id]
            max_similarity = 0.0
            
            for db_embedding in claimed_embeddings:
                db_embedding_np = np.array(db_embedding)
                similarity = self.database._calculate_similarity(
                    embedding_result["embedding"], db_embedding_np
                )
                max_similarity = max(max_similarity, similarity)
            
            is_verified = max_similarity >= self.config.similarity_threshold
            
            total_time = time.time() - start_time
            
            result = {
                "verified": is_verified,
                "claimed_speaker": claimed_speaker_id,
                "similarity": max_similarity,
                "threshold": self.config.similarity_threshold,
                "processing_time": total_time
            }
            
            logger.info(f"Speaker verification completed in {total_time:.3f}s")
            logger.info(f"Verification result: {is_verified} "
                       f"(similarity: {max_similarity:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in speaker verification: {e}")
            return {
                "verified": False,
                "error": str(e)
            }
    
    async def _load_audio(self, 
                         audio_input: Union[str, np.ndarray, torch.Tensor],
                         sample_rate: int = None) -> Tuple[np.ndarray, int]:
        """Load audio from various input types"""
        try:
            if isinstance(audio_input, str):
                if not os.path.exists(audio_input):
                    raise FileNotFoundError(f"Audio file not found: {audio_input}")
                
                audio_array, sr = librosa.load(audio_input, sr=self.config.sample_rate)
                logger.info(f"Loaded audio from {audio_input}, duration: {len(audio_array)/sr:.2f}s")
                
            elif isinstance(audio_input, np.ndarray):
                if sample_rate is None:
                    raise ValueError("sample_rate must be provided for numpy array input")
                audio_array = audio_input
                sr = sample_rate
                
                # Resample if necessary
                if sr != self.config.sample_rate:
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=self.config.sample_rate)
                    sr = self.config.sample_rate
                
            elif isinstance(audio_input, torch.Tensor):
                if sample_rate is None:
                    raise ValueError("sample_rate must be provided for tensor input")
                audio_array = audio_input.cpu().numpy()
                sr = sample_rate
                
                # Resample if necessary
                if sr != self.config.sample_rate:
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=self.config.sample_rate)
                    sr = self.config.sample_rate
                
            else:
                raise ValueError(f"Unsupported audio input type: {type(audio_input)}")
            
            # Ensure mono
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=0)
            
            return audio_array, sr
            
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the speaker database"""
        return {
            "total_speakers": self.database.get_speaker_count(),
            "total_embeddings": sum(len(embeddings) for embeddings in self.database.embeddings_db.values()),
            "database_path": self.config.embedding_db_path
        }

# Utility functions
async def create_speaker_recognizer(config: SpeakerRecogConfig = None) -> SpeakerRecognizer:
    """Factory function to create speaker recognizer"""
    return SpeakerRecognizer(config)

# Example usage
async def main():
    """Example usage of the speaker recognizer"""
    config = SpeakerRecogConfig()
    recognizer = await create_speaker_recognizer(config)
    
    # Register a speaker
    result = await recognizer.register_speaker(
        "path/to/speaker_audio.wav",
        "speaker_001",
        metadata={"name": "John Doe", "gender": "male"}
    )
    
    print(f"Registration: {result['success']}")
    
    # Identify a speaker
    identification = await recognizer.identify_speaker("path/to/unknown_audio.wav")
    print(f"Identified: {identification['is_identified']}")
    if identification['best_match']:
        print(f"Best match: {identification['best_match']['speaker_id']}")

if __name__ == "__main__":
    asyncio.run(main())
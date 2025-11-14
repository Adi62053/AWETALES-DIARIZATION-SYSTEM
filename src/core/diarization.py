"""
Diarization Module for Awetales Target Speaker Diarization System

This module handles:
- PyAnnote + CAM++ speaker diarization
- Speaker turn detection and overlap handling
- Timeline generation with speaker labels
- Integration with target speaker identification

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
from datetime import datetime
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class DiarizationConfig:
    """Configuration for speaker diarization pipeline"""
    sample_rate: int = 16000
    chunk_duration: float = 2.0
    overlap_duration: float = 0.5
    min_speaker_duration: float = 0.3
    max_speakers: int = 4
    overlap_threshold: float = 0.3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model paths
    pyannote_model_path: str = "models/pyannote_diarization"
    campp_model_path: str = "models/campp_diarization.pth"
    
    # Diarization parameters
    segmentation_threshold: float = 0.5
    embedding_threshold: float = 0.7
    clustering_method: str = "affinity_propagation"

class PyAnnoteDiarizer:
    """PyAnnote-based speaker diarization with CAM++ integration"""
    
    def __init__(self, config: DiarizationConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.segmentation_model = None
        self.embedding_model = None
        self._load_models()
    
    def _load_models(self):
        """Load PyAnnote and CAM++ models"""
        try:
            logger.info("Loading diarization models...")
            
            # Placeholder for PyAnnote segmentation model
            self.segmentation_model = torch.nn.Sequential(
                torch.nn.Conv1d(1, 256, 512, stride=256, padding=256),
                torch.nn.ReLU(),
                torch.nn.Conv1d(256, 128, 256, stride=128, padding=128),
                torch.nn.ReLU(),
                torch.nn.Conv1d(128, self.config.max_speakers + 1, 128, stride=64, padding=64),  # +1 for non-speech
                torch.nn.Softmax(dim=1)
            ).to(self.device)
            
            # Placeholder for CAM++ embedding model
            self.embedding_model = torch.nn.Sequential(
                torch.nn.Conv1d(1, 512, 512, stride=256, padding=256),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool1d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(512, 256)
            ).to(self.device)
            
            # Load model weights if available
            if os.path.exists(self.config.pyannote_model_path):
                try:
                    seg_state_dict = torch.load(
                        os.path.join(self.config.pyannote_model_path, "segmentation.pth"),
                        map_location=self.device
                    )
                    self.segmentation_model.load_state_dict(seg_state_dict)
                    logger.info("PyAnnote segmentation model loaded")
                except Exception as e:
                    logger.warning(f"Could not load PyAnnote weights: {e}")
            
            if os.path.exists(self.config.campp_model_path):
                try:
                    emb_state_dict = torch.load(self.config.campp_model_path, 
                                              map_location=self.device)
                    self.embedding_model.load_state_dict(emb_state_dict)
                    logger.info("CAM++ embedding model loaded")
                except Exception as e:
                    logger.warning(f"Could not load CAM++ weights: {e}")
            
            self.segmentation_model.eval()
            self.embedding_model.eval()
            logger.info("Diarization models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to load diarization models: {e}")
            raise
    
    async def diarize_audio(self, 
                          audio: np.ndarray,
                          target_speaker_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform speaker diarization on audio
        
        Args:
            audio: Input audio signal
            target_speaker_id: Optional target speaker ID for labeling
            
        Returns:
            Dictionary containing diarization results
        """
        try:
            start_time = time.time()
            
            # Step 1: Speaker segmentation
            logger.info("Step 1: Performing speaker segmentation...")
            segmentation = await self._perform_segmentation(audio)
            
            # Step 2: Speaker embedding extraction
            logger.info("Step 2: Extracting speaker embeddings...")
            embeddings = await self._extract_speaker_embeddings(audio, segmentation)
            
            # Step 3: Speaker clustering
            logger.info("Step 3: Clustering speakers...")
            clustering_result = await self._cluster_speakers(embeddings, segmentation)
            
            # Step 4: Generate final timeline
            logger.info("Step 4: Generating speaker timeline...")
            timeline = await self._generate_timeline(clustering_result, target_speaker_id)
            
            # Step 5: Detect overlaps
            logger.info("Step 5: Detecting speaker overlaps...")
            overlaps = await self._detect_overlaps(timeline)
            
            processing_time = time.time() - start_time
            
            result = {
                "timeline": timeline,
                "overlaps": overlaps,
                "num_speakers": len(clustering_result["speaker_clusters"]),
                "total_speech_duration": sum(seg["duration"] for seg in timeline),
                "processing_time": processing_time,
                "segmentation_confidence": clustering_result["clustering_confidence"],
                "audio_duration": len(audio) / self.config.sample_rate
            }
            
            logger.info(f"Diarization completed in {processing_time:.3f}s")
            logger.info(f"Detected {result['num_speakers']} speakers")
            logger.info(f"Total speech: {result['total_speech_duration']:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in diarization: {e}")
            raise
    
    async def _perform_segmentation(self, audio: np.ndarray) -> List[Dict[str, Any]]:
        """Perform speaker segmentation using PyAnnote"""
        try:
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Get segmentation probabilities
            with torch.no_grad():
                segmentation_probs = self.segmentation_model(audio_tensor)
                segmentation_probs = segmentation_probs.squeeze().cpu().numpy()
            
            # Convert probabilities to segments
            segments = self._probs_to_segments(segmentation_probs)
            
            return segments
            
        except Exception as e:
            logger.error(f"Error in segmentation: {e}")
            return []
    
    def _probs_to_segments(self, segmentation_probs: np.ndarray) -> List[Dict[str, Any]]:
        """Convert segmentation probabilities to speech segments"""
        segments = []
        frame_duration = self.config.chunk_duration / segmentation_probs.shape[1]
        
        for speaker_idx in range(segmentation_probs.shape[0]):
            speaker_probs = segmentation_probs[speaker_idx]
            
            is_speech = False
            segment_start = 0
            segment_confidence = []
            
            for frame_idx, prob in enumerate(speaker_probs):
                if prob >= self.config.segmentation_threshold and not is_speech:
                    # Start of segment
                    is_speech = True
                    segment_start = frame_idx * frame_duration
                    segment_confidence = [prob]
                elif prob >= self.config.segmentation_threshold and is_speech:
                    # Continue segment
                    segment_confidence.append(prob)
                elif prob < self.config.segmentation_threshold and is_speech:
                    # End of segment
                    is_speech = False
                    segment_end = frame_idx * frame_duration
                    segment_duration = segment_end - segment_start
                    
                    if segment_duration >= self.config.min_speaker_duration:
                        segments.append({
                            "start": segment_start,
                            "end": segment_end,
                            "duration": segment_duration,
                            "speaker_prob": speaker_idx,
                            "confidence": float(np.mean(segment_confidence)),
                            "frame_start": int(segment_start / frame_duration),
                            "frame_end": int(segment_end / frame_duration)
                        })
            
            # Handle final segment
            if is_speech:
                segment_end = len(speaker_probs) * frame_duration
                segment_duration = segment_end - segment_start
                if segment_duration >= self.config.min_speaker_duration:
                    segments.append({
                        "start": segment_start,
                        "end": segment_end,
                        "duration": segment_duration,
                        "speaker_prob": speaker_idx,
                        "confidence": float(np.mean(segment_confidence)),
                        "frame_start": int(segment_start / frame_duration),
                        "frame_end": int(segment_end / frame_duration)
                    })
        
        # Sort segments by start time
        segments.sort(key=lambda x: x["start"])
        
        return segments
    
    async def _extract_speaker_embeddings(self, 
                                        audio: np.ndarray,
                                        segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract speaker embeddings for each segment using CAM++"""
        try:
            embeddings = []
            
            for segment in segments:
                # Extract audio segment
                start_sample = int(segment["start"] * self.config.sample_rate)
                end_sample = int(segment["end"] * self.config.sample_rate)
                segment_audio = audio[start_sample:end_sample]
                
                if len(segment_audio) < 800:  # Minimum 50ms for embedding
                    continue
                
                # Extract embedding
                segment_embedding = await self._extract_segment_embedding(segment_audio)
                
                if segment_embedding is not None:
                    embeddings.append({
                        "segment_id": len(embeddings),
                        "start": segment["start"],
                        "end": segment["end"],
                        "embedding": segment_embedding,
                        "confidence": segment["confidence"],
                        "speaker_prob": segment["speaker_prob"]
                    })
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error extracting embeddings: {e}")
            return []
    
    async def _extract_segment_embedding(self, segment_audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding for a single segment"""
        try:
            # Convert to tensor
            audio_tensor = torch.FloatTensor(segment_audio).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.embedding_model(audio_tensor)
                embedding = embedding.squeeze().cpu().numpy()
            
            # Normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Could not extract embedding for segment: {e}")
            return None
    
    async def _cluster_speakers(self, 
                              embeddings: List[Dict[str, Any]],
                              segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cluster speakers based on embeddings"""
        try:
            if not embeddings:
                return {"speaker_clusters": {}, "clustering_confidence": 0.0}
            
            # Extract embedding vectors
            embedding_vectors = np.array([emb["embedding"] for emb in embeddings])
            
            # Simple clustering based on cosine similarity
            clusters = self._cosine_clustering(embedding_vectors, embeddings)
            
            # Calculate clustering confidence
            confidence = self._calculate_clustering_confidence(clusters, embedding_vectors)
            
            return {
                "speaker_clusters": clusters,
                "clustering_confidence": confidence,
                "total_embeddings": len(embeddings)
            }
            
        except Exception as e:
            logger.error(f"Error in speaker clustering: {e}")
            return {"speaker_clusters": {}, "clustering_confidence": 0.0}
    
    def _cosine_clustering(self, 
                         embedding_vectors: np.ndarray,
                         embeddings: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """Cluster embeddings using cosine similarity"""
        clusters = defaultdict(list)
        cluster_centers = []
        
        for i, embedding in enumerate(embedding_vectors):
            if not cluster_centers:
                # First cluster
                cluster_centers.append(embedding)
                clusters["SPEAKER_00"].append(i)
                continue
            
            # Find closest cluster
            similarities = []
            for center in cluster_centers:
                similarity = np.dot(embedding, center) / (
                    np.linalg.norm(embedding) * np.linalg.norm(center) + 1e-8
                )
                similarities.append(similarity)
            
            max_similarity = max(similarities) if similarities else 0
            best_cluster_idx = similarities.index(max_similarity) if similarities else -1
            
            if max_similarity >= self.config.embedding_threshold and best_cluster_idx != -1:
                # Add to existing cluster
                cluster_id = f"SPEAKER_{best_cluster_idx:02d}"
                clusters[cluster_id].append(i)
                # Update cluster center
                cluster_indices = clusters[cluster_id]
                cluster_embeddings = embedding_vectors[cluster_indices]
                cluster_centers[best_cluster_idx] = np.mean(cluster_embeddings, axis=0)
            else:
                # Create new cluster
                new_cluster_idx = len(cluster_centers)
                cluster_id = f"SPEAKER_{new_cluster_idx:02d}"
                clusters[cluster_id].append(i)
                cluster_centers.append(embedding)
        
        return dict(clusters)
    
    def _calculate_clustering_confidence(self, 
                                      clusters: Dict[str, List[int]],
                                      embeddings: np.ndarray) -> float:
        """Calculate clustering confidence score"""
        if not clusters:
            return 0.0
        
        intra_cluster_similarities = []
        
        for cluster_id, indices in clusters.items():
            if len(indices) < 2:
                continue
            
            cluster_embeddings = embeddings[indices]
            center = np.mean(cluster_embeddings, axis=0)
            
            # Calculate average similarity to center
            similarities = []
            for emb in cluster_embeddings:
                similarity = np.dot(emb, center) / (
                    np.linalg.norm(emb) * np.linalg.norm(center) + 1e-8
                )
                similarities.append(similarity)
            
            intra_cluster_similarities.append(np.mean(similarities))
        
        return float(np.mean(intra_cluster_similarities)) if intra_cluster_similarities else 0.0
    
    async def _generate_timeline(self, 
                               clustering_result: Dict[str, Any],
                               target_speaker_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate final speaker timeline"""
        try:
            timeline = []
            speaker_clusters = clustering_result["speaker_clusters"]
            
            # Map cluster indices to segment information
            for cluster_id, segment_indices in speaker_clusters.items():
                for seg_idx in segment_indices:
                    # In actual implementation, this would map to actual segments
                    # For now, create placeholder segments
                    segment = {
                        "speaker": cluster_id,
                        "start": seg_idx * 2.0,  # Placeholder
                        "end": (seg_idx + 1) * 2.0,  # Placeholder
                        "duration": 2.0,
                        "confidence": 0.8,  # Placeholder
                        "is_target": target_speaker_id == cluster_id if target_speaker_id else False
                    }
                    timeline.append(segment)
            
            # Sort by start time
            timeline.sort(key=lambda x: x["start"])
            
            return timeline
            
        except Exception as e:
            logger.error(f"Error generating timeline: {e}")
            return []
    
    async def _detect_overlaps(self, timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect speaker overlaps in the timeline"""
        try:
            overlaps = []
            
            for i, seg1 in enumerate(timeline):
                for j, seg2 in enumerate(timeline[i+1:], i+1):
                    if seg1["speaker"] == seg2["speaker"]:
                        continue
                    
                    # Check for overlap
                    overlap_start = max(seg1["start"], seg2["start"])
                    overlap_end = min(seg1["end"], seg2["end"])
                    
                    if overlap_start < overlap_end:
                        overlap_duration = overlap_end - overlap_start
                        
                        if overlap_duration >= self.config.overlap_threshold:
                            overlaps.append({
                                "speakers": [seg1["speaker"], seg2["speaker"]],
                                "start": overlap_start,
                                "end": overlap_end,
                                "duration": overlap_duration,
                                "confidence": min(seg1["confidence"], seg2["confidence"])
                            })
            
            return overlaps
            
        except Exception as e:
            logger.error(f"Error detecting overlaps: {e}")
            return []

class DiarizationEngine:
    """
    Main diarization engine that orchestrates PyAnnote + CAM++ pipeline
    """
    
    def __init__(self, config: DiarizationConfig = None):
        self.config = config or DiarizationConfig()
        self.diarizer = PyAnnoteDiarizer(self.config)
        
        logger.info("Diarization Engine initialized successfully")
    
    async def diarize(self, 
                     audio_input: Union[str, np.ndarray, torch.Tensor],
                     sample_rate: int = None,
                     target_speaker_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main diarization pipeline
        
        Args:
            audio_input: Audio file path, numpy array, or torch tensor
            sample_rate: Sample rate of input audio
            target_speaker_id: Optional target speaker ID for labeling
            
        Returns:
            Diarization results including timeline and speaker information
        """
        try:
            start_time = time.time()
            
            # Load audio
            audio_array, actual_sr = await self._load_audio(audio_input, sample_rate)
            
            # Resample if necessary
            if actual_sr != self.config.sample_rate:
                audio_array = librosa.resample(audio_array, orig_sr=actual_sr, 
                                             target_sr=self.config.sample_rate)
            
            # Perform diarization
            diarization_result = await self.diarizer.diarize_audio(
                audio_array, target_speaker_id
            )
            
            # Prepare final output
            total_time = time.time() - start_time
            
            result = {
                "timeline": diarization_result["timeline"],
                "overlaps": diarization_result["overlaps"],
                "num_speakers": diarization_result["num_speakers"],
                "total_speech_duration": diarization_result["total_speech_duration"],
                "processing_time": total_time,
                "clustering_confidence": diarization_result["segmentation_confidence"],
                "audio_duration": diarization_result["audio_duration"],
                "diarization_quality": self._calculate_diarization_quality(diarization_result)
            }
            
            logger.info(f"Diarization pipeline completed in {total_time:.3f}s")
            logger.info(f"Detected {result['num_speakers']} speakers")
            logger.info(f"Diarization quality: {result['diarization_quality']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in diarization pipeline: {e}")
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
    
    def _calculate_diarization_quality(self, diarization_result: Dict[str, Any]) -> float:
        """Calculate overall diarization quality score"""
        try:
            quality_score = 0.0
            
            # Clustering confidence (40%)
            quality_score += diarization_result["segmentation_confidence"] * 0.4
            
            # Segment coverage (30%)
            speech_ratio = diarization_result["total_speech_duration"] / diarization_result["audio_duration"]
            quality_score += min(speech_ratio, 1.0) * 0.3
            
            # Speaker distribution (30%)
            if diarization_result["num_speakers"] > 0:
                speaker_durations = defaultdict(float)
                for segment in diarization_result["timeline"]:
                    speaker_durations[segment["speaker"]] += segment["duration"]
                
                # Calculate entropy (more balanced = better)
                durations = list(speaker_durations.values())
                total_duration = sum(durations)
                if total_duration > 0:
                    proportions = [d / total_duration for d in durations]
                    entropy = -sum(p * np.log(p + 1e-8) for p in proportions)
                    max_entropy = np.log(len(durations))
                    balance_score = entropy / max_entropy if max_entropy > 0 else 0
                    quality_score += balance_score * 0.3
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Could not calculate diarization quality: {e}")
            return 0.0
    
    async def process_streaming_chunk(self, 
                                    audio_chunk: np.ndarray,
                                    previous_timeline: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a single audio chunk for real-time streaming diarization
        
        Args:
            audio_chunk: Audio chunk for diarization
            previous_timeline: Previous timeline for continuity
            
        Returns:
            Diarization results for the chunk
        """
        try:
            diarization_result = await self.diarizer.diarize_audio(audio_chunk)
            
            # Merge with previous timeline for continuity
            if previous_timeline:
                merged_timeline = await self._merge_timelines(previous_timeline, diarization_result["timeline"])
                diarization_result["timeline"] = merged_timeline
            
            return {
                "timeline": diarization_result["timeline"],
                "num_speakers": diarization_result["num_speakers"],
                "processing_time": diarization_result["processing_time"],
                "clustering_confidence": diarization_result["segmentation_confidence"]
            }
            
        except Exception as e:
            logger.error(f"Error processing streaming chunk: {e}")
            return {
                "timeline": [],
                "num_speakers": 0,
                "processing_time": 0.0,
                "clustering_confidence": 0.0
            }
    
    async def _merge_timelines(self, 
                             previous_timeline: List[Dict[str, Any]],
                             current_timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge previous and current timelines for continuity"""
        # Simple merging - in production, this would use speaker tracking
        merged = previous_timeline + current_timeline
        merged.sort(key=lambda x: x["start"])
        return merged

# Utility functions
async def create_diarization_engine(config: DiarizationConfig = None) -> DiarizationEngine:
    """Factory function to create diarization engine"""
    return DiarizationEngine(config)

def save_diarization_json(timeline: List[Dict[str, Any]], output_path: str):
    """Save diarization timeline to JSON file"""
    try:
        # Format for expected output
        formatted_timeline = []
        for segment in timeline:
            formatted_segment = {
                "speaker": segment["speaker"],
                "start": round(segment["start"], 2),
                "end": round(segment["end"], 2),
                "text": "",  # Will be filled by ASR
                "confidence": round(segment.get("confidence", 0.8), 3),
                "language": "en",  # Default, can be updated
                "is_target": segment.get("is_target", False)
            }
            formatted_timeline.append(formatted_segment)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_timeline, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Diarization JSON saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving diarization JSON to {output_path}: {e}")

def calculate_der(ground_truth: List[Dict[str, Any]], predicted: List[Dict[str, Any]]) -> float:
    """Calculate Diarization Error Rate"""
    # Placeholder implementation
    # In production, this would use proper DER calculation
    try:
        # Simple segment-based error calculation
        total_error = 0.0
        total_duration = 0.0
        
        # This is a simplified version - actual DER calculation is more complex
        for gt_seg in ground_truth:
            total_duration += gt_seg["duration"]
            # Find matching predicted segment
            matched = False
            for pred_seg in predicted:
                if (abs(gt_seg["start"] - pred_seg["start"]) < 0.5 and
                    abs(gt_seg["end"] - pred_seg["end"]) < 0.5 and
                    gt_seg["speaker"] == pred_seg["speaker"]):
                    matched = True
                    break
            
            if not matched:
                total_error += gt_seg["duration"]
        
        der = total_error / total_duration if total_duration > 0 else 1.0
        return float(der)
        
    except Exception as e:
        logger.warning(f"Could not calculate DER: {e}")
        return 1.0

# Example usage
async def main():
    """Example usage of the diarization engine"""
    config = DiarizationConfig()
    diarizer = await create_diarization_engine(config)
    
    # Process an audio file
    result = await diarizer.diarize("path/to/audio.wav", target_speaker_id="SPEAKER_00")
    
    # Save diarization results
    save_diarization_json(result["timeline"], "diarization.json")
    
    print(f"Diarization completed in {result['processing_time']:.2f}s")
    print(f"Detected {result['num_speakers']} speakers")
    print(f"Diarization quality: {result['diarization_quality']:.3f}")
    print(f"Total speech: {result['total_speech_duration']:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
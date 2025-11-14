from fastapi import APIRouter, HTTPException, WebSocket, Query, Path, BackgroundTasks, File, UploadFile
from typing import Optional, List, Dict, Any
import logging
import uuid
from pydantic import BaseModel, Field
from datetime import datetime

logger = logging.getLogger(__name__)

# Create router
api_router = APIRouter()

# Request/Response models
class ProcessingRequest(BaseModel):
    """Base model for audio processing requests"""
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    target_speaker_embedding: Optional[str] = None
    enable_denoising: bool = True

class ProcessingResponse(BaseModel):
    """Base model for processing responses"""
    session_id: str
    status: str
    message: str

class ProcessingResultResponse(BaseModel):
    """Model for complete processing results"""
    session_id: str
    status: str
    processing_time: float
    output_files: Dict[str, str]

class BatchProcessingRequest(BaseModel):
    """Model for batch processing requests"""
    audio_urls: List[str]
    config: Dict[str, Any] = {}

# Health endpoint
@api_router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "system": "Awetales Diarization System",
        "version": "1.0.0"
    }

# Metrics endpoint
@api_router.get("/metrics")
async def get_metrics():
    return {
        "status": "operational",
        "active_connections": 0,
        "system_health": "excellent"
    }

# Process audio endpoint
@api_router.post("/process", response_model=ProcessingResponse)
async def process_audio(
    background_tasks: BackgroundTasks,
    target_speaker_embedding: Optional[str] = Query(None),
    enable_denoising: bool = Query(True)
):
    """Process audio file for diarization and transcription"""
    try:
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        logger.info(f"Starting audio processing for session: {session_id}")
        
        # Mock processing for now
        return ProcessingResponse(
            session_id=session_id,
            status="processing",
            message="Audio processing started successfully"
        )
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get processing result - FIXED: Use Path for path parameter
@api_router.get("/process/{session_id}", response_model=ProcessingResultResponse)
async def get_processing_result(
    session_id: str = Path(..., description="Session ID to check results")
):
    """Get processing result by session ID"""
    try:
        # Mock results for now
        return ProcessingResultResponse(
            session_id=session_id,
            status="completed",
            processing_time=2.5,
            output_files={
                "target_speaker": f"data/output/{session_id}_speaker.wav",
                "transcript": f"data/output/{session_id}_diarization.json",
                "metrics": f"data/output/{session_id}_metrics.json"
            }
        )
    except Exception as e:
        logger.error(f"Error getting result: {e}")
        raise HTTPException(status_code=404, detail="Session not found")

# Download target audio - FIXED: Use Path for path parameter
@api_router.get("/process/{session_id}/audio")
async def download_target_audio(
    session_id: str = Path(..., description="Session ID to download target speaker audio")
):
    """Download target speaker audio"""
    try:
        # Mock implementation
        return {
            "session_id": session_id,
            "audio_url": f"/api/v1/process/{session_id}/download",
            "message": "Audio file ready for download"
        }
    except Exception as e:
        logger.error(f"Error downloading audio: {e}")
        raise HTTPException(status_code=404, detail="Audio not found")

# Cancel processing - FIXED: Use Path for path parameter
@api_router.delete("/process/{session_id}")
async def cancel_processing(
    session_id: str = Path(..., description="Session ID to cancel")
):
    """Cancel ongoing processing"""
    try:
        return {
            "session_id": session_id,
            "status": "cancelled",
            "message": "Processing cancelled successfully"
        }
    except Exception as e:
        logger.error(f"Error cancelling processing: {e}")
        raise HTTPException(status_code=404, detail="Session not found")

# Batch processing endpoint
@api_router.post("/process/batch")
async def process_batch_audio(batch_request: BatchProcessingRequest):
    """Process multiple audio files in batch"""
    try:
        results = []
        for i, audio_url in enumerate(batch_request.audio_urls):
            session_id = f"batch_{uuid.uuid4()}_{i}"
            results.append({
                "session_id": session_id,
                "status": "completed",
                "audio_url": audio_url,
                "output_files": {
                    "target_speaker": f"data/output/{session_id}_speaker.wav",
                    "transcript": f"data/output/{session_id}_diarization.json"
                }
            })
        
        return {
            "batch_id": str(uuid.uuid4()),
            "total_files": len(batch_request.audio_urls),
            "results": results
        }
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time streaming
@api_router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back for testing
            await websocket.send_json({
                "status": "received",
                "message": "WebSocket connection active",
                "data": data
            })
    except Exception as e:
        logger.info(f"WebSocket closed: {e}")

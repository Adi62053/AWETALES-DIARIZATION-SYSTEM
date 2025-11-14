"""
Metrics Calculator for Awetales Diarization System

Computes WER, DER, SI-SDR, precision, and other quality metrics
for audio processing and speaker diarization evaluation.
"""

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
import threading
from collections import defaultdict, deque
import scipy.signal
import scipy.fft
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Optional imports with fallbacks
try:
    import Levenshtein
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False
    Levenshtein = None
    logger = logging.getLogger(__name__)
    logger.warning("Levenshtein package not available. Using fallback WER calculation.")

try:
    from dataclasses_json import dataclass_json
    DATACLASSES_JSON_AVAILABLE = True
except ImportError:
    DATACLASSES_JSON_AVAILABLE = False
    # Create a dummy decorator if dataclasses_json is not available
    def dataclass_json(cls):
        return cls
    logger.warning("dataclasses_json package not available. JSON serialization limited.")

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Metric type enumeration"""
    WER = "wer"  # Word Error Rate
    DER = "der"  # Diarization Error Rate
    SI_SDR = "si_sdr"  # Scale-Invariant Signal-to-Distortion Ratio
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    CONFIDENCE = "confidence"

@dataclass_json
@dataclass
class WordAlignment:
    """Word alignment result for WER calculation"""
    reference: str
    hypothesis: str
    alignment: List[Tuple[str, str]]  # (ref_word, hyp_word) pairs
    substitutions: int
    deletions: int
    insertions: int

@dataclass_json
@dataclass
class DiarizationSegment:
    """Diarization segment for DER calculation"""
    start_time: float
    end_time: float
    speaker_id: str
    confidence: float = 1.0

@dataclass_json
@dataclass
class QualityMetrics:
    """Comprehensive quality metrics"""
    session_id: str
    timestamp: float
    wer: float = 0.0
    der: float = 0.0
    si_sdr: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    confidence: float = 0.0
    word_alignment: Optional[WordAlignment] = None
    diarization_segments: List[DiarizationSegment] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass_json
@dataclass
class StatisticalSummary:
    """Statistical summary of metrics"""
    mean: float
    std: float
    min: float
    max: float
    median: float
    confidence_interval: Tuple[float, float]  # 95% CI
    sample_size: int

@dataclass_json
@dataclass
class MetricThresholds:
    """Quality metric thresholds"""
    wer_warning: float = 0.15
    wer_critical: float = 0.25
    der_warning: float = 0.10
    der_critical: float = 0.20
    si_sdr_warning: float = 6.0
    si_sdr_critical: float = 4.0
    precision_warning: float = 0.85
    precision_critical: float = 0.75

class MetricsCalculator:
    """
    Comprehensive quality metrics calculator for audio processing evaluation.
    
    Features:
    - Word Error Rate (WER) calculation with alignment
    - Diarization Error Rate (DER) computation
    - Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
    - Precision, recall, F1-score for speaker identification
    - Statistical analysis and confidence intervals
    - Real-time incremental updates
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MetricsCalculator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = self._load_config(config)
        
        # Metrics storage
        self.metrics_history: Dict[str, List[QualityMetrics]] = defaultdict(list)
        self.ground_truth_data: Dict[str, Any] = {}
        
        # Statistical state
        self.metric_summaries: Dict[MetricType, StatisticalSummary] = {}
        self.trend_analysis: Dict[MetricType, Dict[str, float]] = {}
        
        # Threading
        self.calculation_lock = threading.RLock()
        self.thread_pool = None
        
        # Thresholds
        self.thresholds = MetricThresholds()
        
        logger.info("MetricsCalculator initialized with config: %s", self.config)
    
    def _load_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load and validate configuration"""
        default_config = {
            "wer_collar": 0.25,  # 250ms collar for DER
            "sample_rate": 16000,
            "confidence_level": 0.95,
            "bootstrap_samples": 1000,
            "max_history_size": 1000,
            "enable_real_time": True,
            "language": "en",
            "tokenization_method": "whitespace"
        }
        
        if config:
            default_config.update(config)
        
        return default_config
    
    def calculate_wer(self, reference: str, hypothesis: str, 
                     language: Optional[str] = None) -> Tuple[float, WordAlignment]:
        """
        Calculate Word Error Rate with detailed alignment.
        
        Args:
            reference: Reference transcript
            hypothesis: Hypothesis transcript from ASR
            language: Language for tokenization (default: config language)
            
        Returns:
            Tuple[float, WordAlignment]: WER and alignment details
        """
        start_time = time.time()
        
        try:
            lang = language or self.config["language"]
            tokenization = self.config["tokenization_method"]
            
            # Tokenize based on language and method
            ref_tokens = self._tokenize_text(reference, lang, tokenization)
            hyp_tokens = self._tokenize_text(hypothesis, lang, tokenization)
            
            # Calculate edit distance and alignment
            if LEVENSHTEIN_AVAILABLE:
                distance = Levenshtein.distance(ref_tokens, hyp_tokens)
                operations = Levenshtein.editops(ref_tokens, hyp_tokens)
            else:
                # Fallback implementation using dynamic programming
                distance, operations = self._levenshtein_fallback(ref_tokens, hyp_tokens)
            
            # Count operation types
            substitutions = sum(1 for op in operations if op[0] == 'replace')
            deletions = sum(1 for op in operations if op[0] == 'delete')
            insertions = sum(1 for op in operations if op[0] == 'insert')
            
            # Calculate WER
            total_words = len(ref_tokens)
            if total_words == 0:
                wer = 1.0 if len(hyp_tokens) > 0 else 0.0
            else:
                wer = (substitutions + deletions + insertions) / total_words
            
            # Create alignment
            alignment = self._create_word_alignment(ref_tokens, hyp_tokens, operations)
            
            word_alignment = WordAlignment(
                reference=reference,
                hypothesis=hypothesis,
                alignment=alignment,
                substitutions=substitutions,
                deletions=deletions,
                insertions=insertions
            )
            
            calculation_time = time.time() - start_time
            logger.debug("WER calculation completed in %.3fms: %.4f", 
                        calculation_time * 1000, wer)
            
            return wer, word_alignment
            
        except Exception as e:
            logger.error("Error calculating WER: %s", str(e))
            # Return maximum error rate on failure
            return 1.0, WordAlignment(
                reference=reference,
                hypothesis=hypothesis,
                alignment=[],
                substitutions=0,
                deletions=0,
                insertions=0
            )
    
    def _levenshtein_fallback(self, ref_tokens: List[str], hyp_tokens: List[str]) -> Tuple[int, List]:
        """
        Fallback Levenshtein distance implementation using dynamic programming.
        
        Args:
            ref_tokens: Reference tokens
            hyp_tokens: Hypothesis tokens
            
        Returns:
            Tuple[int, List]: Distance and edit operations
        """
        m, n = len(ref_tokens), len(hyp_tokens)
        
        # Create distance matrix
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize first row and column
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill distance matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i-1] == hyp_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,    # deletion
                        dp[i][j-1] + 1,    # insertion
                        dp[i-1][j-1] + 1   # substitution
                    )
        
        # Backtrack to find operations
        operations = []
        i, j = m, n
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref_tokens[i-1] == hyp_tokens[j-1]:
                # Match
                i -= 1
                j -= 1
            else:
                if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                    # Substitution
                    operations.append(('replace', i-1, j-1))
                    i -= 1
                    j -= 1
                elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                    # Deletion
                    operations.append(('delete', i-1))
                    i -= 1
                else:
                    # Insertion
                    operations.append(('insert', j-1))
                    j -= 1
        
        operations.reverse()
        return dp[m][n], operations
    
    def _tokenize_text(self, text: str, language: str, method: str) -> List[str]:
        """Tokenize text based on language and method"""
        text = text.strip().lower()
        
        if method == "whitespace":
            # Simple whitespace tokenization
            tokens = text.split()
        elif method == "punctuation_aware":
            # More sophisticated tokenization with punctuation handling
            import re
            tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        else:
            tokens = text.split()
        
        # Language-specific processing
        if language in ["zh", "ja", "ko"]:
            # For languages without word boundaries, character-level tokenization
            tokens = list(text.replace(" ", ""))
        
        return tokens
    
    def _create_word_alignment(self, ref_tokens: List[str], hyp_tokens: List[str], 
                              operations: List) -> List[Tuple[str, str]]:
        """Create word alignment from edit operations"""
        alignment = []
        ref_idx = 0
        hyp_idx = 0
        
        # Convert operations to a more usable format
        op_dict = {}
        for op in operations:
            if op[0] == 'replace':
                op_dict[('replace', op[1], op[2])] = True
            elif op[0] == 'delete':
                op_dict[('delete', op[1])] = True
            elif op[0] == 'insert':
                op_dict[('insert', op[2])] = True
        
        while ref_idx < len(ref_tokens) or hyp_idx < len(hyp_tokens):
            ref_word = ref_tokens[ref_idx] if ref_idx < len(ref_tokens) else ""
            hyp_word = hyp_tokens[hyp_idx] if hyp_idx < len(hyp_tokens) else ""
            
            if ('replace', ref_idx, hyp_idx) in op_dict:
                alignment.append((ref_word, hyp_word))
                ref_idx += 1
                hyp_idx += 1
            elif ('delete', ref_idx) in op_dict:
                alignment.append((ref_word, ""))
                ref_idx += 1
            elif ('insert', hyp_idx) in op_dict:
                alignment.append(("", hyp_word))
                hyp_idx += 1
            else:
                alignment.append((ref_word, hyp_word))
                ref_idx += 1
                hyp_idx += 1
        
        return alignment
    
    def calculate_der(self, reference_segments: List[DiarizationSegment],
                     hypothesis_segments: List[DiarizationSegment],
                     collar: Optional[float] = None) -> float:
        """
        Calculate Diarization Error Rate with collar-based evaluation.
        
        Args:
            reference_segments: Ground truth diarization segments
            hypothesis_segments: System output diarization segments
            collar: Collar size in seconds (default: config value)
            
        Returns:
            float: DER value
        """
        start_time = time.time()
        
        try:
            collar_size = collar or self.config["wer_collar"]
            
            if not reference_segments or not hypothesis_segments:
                return 1.0  # Maximum error if no segments
            
            # Convert segments to evaluation format
            ref_segments = self._prepare_segments_for_der(reference_segments, collar_size)
            hyp_segments = self._prepare_segments_for_der(hypothesis_segments, collar_size)
            
            # Calculate DER using NIST md-eval style approach
            total_ref_time = sum(seg.end_time - seg.start_time for seg in ref_segments)
            
            if total_ref_time == 0:
                return 1.0
            
            # Calculate error components
            speaker_error = self._calculate_speaker_error(ref_segments, hyp_segments)
            false_alarm_error = self._calculate_false_alarm_error(ref_segments, hyp_segments)
            missed_speech_error = self._calculate_missed_speech_error(ref_segments, hyp_segments)
            
            der = (speaker_error + false_alarm_error + missed_speech_error) / total_ref_time
            
            calculation_time = time.time() - start_time
            logger.debug("DER calculation completed in %.3fms: %.4f", 
                        calculation_time * 1000, der)
            
            return max(0.0, min(der, 1.0))  # Clamp between 0 and 1
            
        except Exception as e:
            logger.error("Error calculating DER: %s", str(e))
            return 1.0  # Maximum error on failure
    
    def _prepare_segments_for_der(self, segments: List[DiarizationSegment], 
                                 collar: float) -> List[DiarizationSegment]:
        """Prepare segments for DER calculation with collar application"""
        processed_segments = []
        
        for segment in segments:
            # Apply collar - shrink segments by collar/2 on each side
            start_time = max(0.0, segment.start_time + collar / 2)
            end_time = max(start_time, segment.end_time - collar / 2)
            
            if end_time > start_time:  # Only include non-zero length segments
                processed_segments.append(DiarizationSegment(
                    start_time=start_time,
                    end_time=end_time,
                    speaker_id=segment.speaker_id,
                    confidence=segment.confidence
                ))
        
        return processed_segments
    
    def _calculate_speaker_error(self, ref_segments: List[DiarizationSegment],
                                hyp_segments: List[DiarizationSegment]) -> float:
        """Calculate speaker misassignment error"""
        error_time = 0.0
        
        # Create time-sorted merged segments
        all_events = []
        for seg in ref_segments:
            all_events.append(('start', seg.start_time, 'ref', seg.speaker_id))
            all_events.append(('end', seg.end_time, 'ref', seg.speaker_id))
        for seg in hyp_segments:
            all_events.append(('start', seg.start_time, 'hyp', seg.speaker_id))
            all_events.append(('end', seg.end_time, 'hyp', seg.speaker_id))
        
        all_events.sort(key=lambda x: x[1])
        
        # Track active segments
        active_ref = defaultdict(list)
        active_hyp = defaultdict(list)
        current_time = 0.0
        
        for event_type, event_time, source, speaker_id in all_events:
            # Calculate error for the time segment just passed
            if event_time > current_time:
                segment_duration = event_time - current_time
                
                # Check for speaker errors in this segment
                if active_ref and active_hyp:
                    # Simple mapping: assume best matching speaker
                    ref_speakers = set(sid for segs in active_ref.values() for sid in segs)
                    hyp_speakers = set(sid for segs in active_hyp.values() for sid in segs)
                    
                    if ref_speakers != hyp_speakers:
                        error_time += segment_duration
            
            # Update active segments
            if source == 'ref':
                if event_type == 'start':
                    active_ref[speaker_id].append(event_time)
                else:  # 'end'
                    if active_ref[speaker_id]:
                        active_ref[speaker_id].pop()
            else:  # 'hyp'
                if event_type == 'start':
                    active_hyp[speaker_id].append(event_time)
                else:  # 'end'
                    if active_hyp[speaker_id]:
                        active_hyp[speaker_id].pop()
            
            current_time = event_time
        
        return error_time
    
    def _calculate_false_alarm_error(self, ref_segments: List[DiarizationSegment],
                                   hyp_segments: List[DiarizationSegment]) -> float:
        """Calculate false alarm (non-speech detected as speech) error"""
        # For simplicity, calculate non-overlapping hypothesis time
        total_hyp_time = sum(seg.end_time - seg.start_time for seg in hyp_segments)
        overlapping_time = self._calculate_overlapping_time(ref_segments, hyp_segments)
        
        return max(0.0, total_hyp_time - overlapping_time)
    
    def _calculate_missed_speech_error(self, ref_segments: List[DiarizationSegment],
                                     hyp_segments: List[DiarizationSegment]) -> float:
        """Calculate missed speech (speech not detected) error"""
        # For simplicity, calculate non-overlapping reference time
        total_ref_time = sum(seg.end_time - seg.start_time for seg in ref_segments)
        overlapping_time = self._calculate_overlapping_time(ref_segments, hyp_segments)
        
        return max(0.0, total_ref_time - overlapping_time)
    
    def _calculate_overlapping_time(self, segments1: List[DiarizationSegment],
                                  segments2: List[DiarizationSegment]) -> float:
        """Calculate total overlapping time between two segment lists"""
        total_overlap = 0.0
        
        for seg1 in segments1:
            for seg2 in segments2:
                overlap_start = max(seg1.start_time, seg2.start_time)
                overlap_end = min(seg1.end_time, seg2.end_time)
                
                if overlap_end > overlap_start:
                    total_overlap += overlap_end - overlap_start
        
        return total_overlap
    
    def calculate_si_sdr(self, reference_audio: np.ndarray, 
                        enhanced_audio: np.ndarray,
                        sample_rate: Optional[int] = None) -> float:
        """
        Calculate Scale-Invariant Signal-to-Distortion Ratio.
        
        Args:
            reference_audio: Original reference audio signal
            enhanced_audio: Enhanced/processed audio signal
            sample_rate: Audio sample rate (default: config value)
            
        Returns:
            float: SI-SDR value in dB
        """
        start_time = time.time()
        
        try:
            sr = sample_rate or self.config["sample_rate"]
            
            # Ensure same length
            min_len = min(len(reference_audio), len(enhanced_audio))
            reference_audio = reference_audio[:min_len]
            enhanced_audio = enhanced_audio[:min_len]
            
            if min_len == 0:
                return -np.inf  # Invalid audio
            
            # Remove DC offset
            reference_audio = reference_audio - np.mean(reference_audio)
            enhanced_audio = enhanced_audio - np.mean(enhanced_audio)
            
            # Calculate scale factor
            scale = np.dot(reference_audio, enhanced_audio) / np.dot(reference_audio, reference_audio)
            
            # Calculate scaled target and error
            target = scale * reference_audio
            error = enhanced_audio - target
            
            # Calculate energies
            target_energy = np.dot(target, target)
            error_energy = np.dot(error, error)
            
            if error_energy == 0:
                return np.inf  # Perfect reconstruction
            if target_energy == 0:
                return -np.inf  # No signal
            
            si_sdr = 10 * np.log10(target_energy / error_energy)
            
            calculation_time = time.time() - start_time
            logger.debug("SI-SDR calculation completed in %.3fms: %.2f dB", 
                        calculation_time * 1000, si_sdr)
            
            return float(si_sdr)
            
        except Exception as e:
            logger.error("Error calculating SI-SDR: %s", str(e))
            return -np.inf  # Minimum value on error
    
    def calculate_speaker_metrics(self, reference_labels: List[str],
                                hypothesis_labels: List[str],
                                confidence_scores: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Calculate speaker identification metrics.
        
        Args:
            reference_labels: Ground truth speaker labels
            hypothesis_labels: Predicted speaker labels
            confidence_scores: Optional confidence scores for predictions
            
        Returns:
            Dict[str, float]: Dictionary of precision, recall, F1-score
        """
        try:
            if not reference_labels or not hypothesis_labels:
                return {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "confidence": 0.0
                }
            
            # Ensure same length
            min_len = min(len(reference_labels), len(hypothesis_labels))
            ref_labels = reference_labels[:min_len]
            hyp_labels = hypothesis_labels[:min_len]
            
            # Calculate basic metrics
            precision = precision_score(ref_labels, hyp_labels, average='weighted', zero_division=0)
            recall = recall_score(ref_labels, hyp_labels, average='weighted', zero_division=0)
            f1 = f1_score(ref_labels, hyp_labels, average='weighted', zero_division=0)
            
            # Calculate confidence
            if confidence_scores and len(confidence_scores) >= min_len:
                conf_scores = confidence_scores[:min_len]
                confidence = float(np.mean(conf_scores))
            else:
                # Estimate confidence from accuracy
                accuracy = np.mean(np.array(ref_labels) == np.array(hyp_labels))
                confidence = accuracy
            
            return {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "confidence": float(confidence)
            }
            
        except Exception as e:
            logger.error("Error calculating speaker metrics: %s", str(e))
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "confidence": 0.0
            }
    
    def calculate_comprehensive_metrics(self, session_id: str,
                                      reference_data: Dict[str, Any],
                                      hypothesis_data: Dict[str, Any]) -> QualityMetrics:
        """
        Calculate comprehensive quality metrics for a session.
        
        Args:
            session_id: Session identifier
            reference_data: Ground truth data
            hypothesis_data: System output data
            
        Returns:
            QualityMetrics: Comprehensive quality metrics
        """
        start_time = time.time()
        
        try:
            metrics = QualityMetrics(
                session_id=session_id,
                timestamp=time.time()
            )
            
            # Calculate WER if transcripts available
            if 'transcript' in reference_data and 'transcript' in hypothesis_data:
                wer, alignment = self.calculate_wer(
                    reference_data['transcript'],
                    hypothesis_data['transcript']
                )
                metrics.wer = wer
                metrics.word_alignment = alignment
            
            # Calculate DER if diarization segments available
            if 'diarization_segments' in reference_data and 'diarization_segments' in hypothesis_data:
                der = self.calculate_der(
                    reference_data['diarization_segments'],
                    hypothesis_data['diarization_segments']
                )
                metrics.der = der
                metrics.diarization_segments = hypothesis_data['diarization_segments']
            
            # Calculate SI-SDR if audio available
            if 'audio' in reference_data and 'audio' in hypothesis_data:
                si_sdr = self.calculate_si_sdr(
                    reference_data['audio'],
                    hypothesis_data['audio']
                )
                metrics.si_sdr = si_sdr
            
            # Calculate speaker metrics if labels available
            if 'speaker_labels' in reference_data and 'speaker_labels' in hypothesis_data:
                speaker_metrics = self.calculate_speaker_metrics(
                    reference_data['speaker_labels'],
                    hypothesis_data['speaker_labels'],
                    hypothesis_data.get('confidence_scores')
                )
                metrics.precision = speaker_metrics['precision']
                metrics.recall = speaker_metrics['recall']
                metrics.f1_score = speaker_metrics['f1_score']
                metrics.confidence = speaker_metrics['confidence']
            
            # Store metadata
            metrics.metadata = {
                "calculation_time": time.time() - start_time,
                "reference_source": reference_data.get('source', 'unknown'),
                "hypothesis_source": hypothesis_data.get('source', 'unknown'),
                "audio_duration": reference_data.get('duration', 0.0)
            }
            
            # Store in history
            with self.calculation_lock:
                self.metrics_history[session_id].append(metrics)
                
                # Limit history size
                if len(self.metrics_history[session_id]) > self.config["max_history_size"]:
                    self.metrics_history[session_id] = self.metrics_history[session_id][-self.config["max_history_size"]:]
            
            logger.info("Comprehensive metrics calculated for session %s: "
                       "WER=%.3f, DER=%.3f, SI-SDR=%.1f dB", 
                       session_id, metrics.wer, metrics.der, metrics.si_sdr)
            
            return metrics
            
        except Exception as e:
            logger.error("Error calculating comprehensive metrics for session %s: %s", 
                        session_id, str(e))
            return QualityMetrics(session_id=session_id, timestamp=time.time())
    
    def calculate_statistical_summary(self, session_id: Optional[str] = None,
                                    metric_type: Optional[MetricType] = None) -> Dict[MetricType, StatisticalSummary]:
        """
        Calculate statistical summary of metrics.
        
        Args:
            session_id: Optional session filter
            metric_type: Optional metric type filter
            
        Returns:
            Dict[MetricType, StatisticalSummary]: Statistical summaries
        """
        try:
            summaries = {}
            metric_values = self._collect_metric_values(session_id, metric_type)
            
            for mtype, values in metric_values.items():
                if not values:
                    continue
                
                values_array = np.array(values)
                n = len(values_array)
                
                # Calculate basic statistics
                mean = float(np.mean(values_array))
                std = float(np.std(values_array))
                min_val = float(np.min(values_array))
                max_val = float(np.max(values_array))
                median = float(np.median(values_array))
                
                # Calculate confidence interval using bootstrap
                ci_lower, ci_upper = self._bootstrap_confidence_interval(values_array)
                
                summaries[mtype] = StatisticalSummary(
                    mean=mean,
                    std=std,
                    min=min_val,
                    max=max_val,
                    median=median,
                    confidence_interval=(ci_lower, ci_upper),
                    sample_size=n
                )
            
            return summaries
            
        except Exception as e:
            logger.error("Error calculating statistical summary: %s", str(e))
            return {}
    
    def _collect_metric_values(self, session_id: Optional[str] = None,
                             metric_type: Optional[MetricType] = None) -> Dict[MetricType, List[float]]:
        """Collect metric values for statistical analysis"""
        metric_values = defaultdict(list)
        
        with self.calculation_lock:
            if session_id:
                # Specific session
                if session_id in self.metrics_history:
                    for metrics in self.metrics_history[session_id]:
                        self._add_metric_values(metrics, metric_values, metric_type)
            else:
                # All sessions
                for session_metrics in self.metrics_history.values():
                    for metrics in session_metrics:
                        self._add_metric_values(metrics, metric_values, metric_type)
        
        return metric_values
    
    def _add_metric_values(self, metrics: QualityMetrics,
                          metric_values: Dict[MetricType, List[float]],
                          filter_type: Optional[MetricType] = None):
        """Add metric values to collection"""
        metric_mapping = {
            MetricType.WER: metrics.wer,
            MetricType.DER: metrics.der,
            MetricType.SI_SDR: metrics.si_sdr,
            MetricType.PRECISION: metrics.precision,
            MetricType.RECALL: metrics.recall,
            MetricType.F1_SCORE: metrics.f1_score,
            MetricType.CONFIDENCE: metrics.confidence
        }
        
        for mtype, value in metric_mapping.items():
            if filter_type is None or mtype == filter_type:
                if value is not None and not np.isnan(value) and not np.isinf(value):
                    metric_values[mtype].append(value)
    
    def _bootstrap_confidence_interval(self, data: np.ndarray, 
                                     confidence_level: float = 0.95,
                                     n_samples: int = 1000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        if len(data) < 2:
            return (float(data[0]), float(data[0])) if len(data) == 1 else (0.0, 0.0)
        
        bootstrap_means = []
        n = len(data)
        
        for _ in range(n_samples):
            # Sample with replacement
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        # Calculate confidence interval
        alpha = (1 - confidence_level) / 2
        lower = np.percentile(bootstrap_means, alpha * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
        
        return float(lower), float(upper)
    
    def detect_metric_trends(self, session_id: str, 
                           metric_type: MetricType,
                           window_size: int = 10) -> Dict[str, float]:
        """
        Detect trends in metric values over time.
        
        Args:
            session_id: Session identifier
            metric_type: Metric type to analyze
            window_size: Window size for trend analysis
            
        Returns:
            Dict[str, float]: Trend analysis results
        """
        try:
            if session_id not in self.metrics_history:
                return {"trend": 0.0, "significance": 0.0}
            
            metrics = self.metrics_history[session_id]
            if len(metrics) < window_size:
                return {"trend": 0.0, "significance": 0.0}
            
            # Extract metric values
            values = []
            for metric in metrics[-window_size:]:
                value = getattr(metric, metric_type.value)
                if value is not None and not np.isnan(value):
                    values.append(value)
            
            if len(values) < 2:
                return {"trend": 0.0, "significance": 0.0}
            
            # Calculate linear trend
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)
            
            # Calculate trend significance (R-squared)
            y_pred = slope * x + intercept
            ss_res = np.sum((values - y_pred) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                "trend": float(slope),
                "significance": float(r_squared),
                "current_value": float(values[-1]),
                "window_size": len(values)
            }
            
        except Exception as e:
            logger.error("Error detecting metric trends: %s", str(e))
            return {"trend": 0.0, "significance": 0.0}
    
    def validate_against_ground_truth(self, session_id: str,
                                    hypothesis_metrics: QualityMetrics) -> Dict[str, Any]:
        """
        Validate metrics against ground truth data.
        
        Args:
            session_id: Session identifier
            hypothesis_metrics: Calculated metrics to validate
            
        Returns:
            Dict[str, Any]: Validation results
        """
        try:
            if session_id not in self.ground_truth_data:
                return {"valid": False, "error": "No ground truth data available"}
            
            ground_truth = self.ground_truth_data[session_id]
            validation_results = {}
            
            # Validate WER
            if hasattr(ground_truth, 'wer') and hypothesis_metrics.wer is not None:
                wer_diff = abs(ground_truth.wer - hypothesis_metrics.wer)
                validation_results["wer_validation"] = {
                    "ground_truth": ground_truth.wer,
                    "calculated": hypothesis_metrics.wer,
                    "difference": wer_diff,
                    "within_tolerance": wer_diff < 0.05  # 5% tolerance
                }
            
            # Validate DER
            if hasattr(ground_truth, 'der') and hypothesis_metrics.der is not None:
                der_diff = abs(ground_truth.der - hypothesis_metrics.der)
                validation_results["der_validation"] = {
                    "ground_truth": ground_truth.der,
                    "calculated": hypothesis_metrics.der,
                    "difference": der_diff,
                    "within_tolerance": der_diff < 0.03  # 3% tolerance
                }
            
            # Validate SI-SDR
            if hasattr(ground_truth, 'si_sdr') and hypothesis_metrics.si_sdr is not None:
                sisdr_diff = abs(ground_truth.si_sdr - hypothesis_metrics.si_sdr)
                validation_results["si_sdr_validation"] = {
                    "ground_truth": ground_truth.si_sdr,
                    "calculated": hypothesis_metrics.si_sdr,
                    "difference": sisdr_diff,
                    "within_tolerance": sisdr_diff < 1.0  # 1 dB tolerance
                }
            
            validation_results["valid"] = all(
                result.get("within_tolerance", True)
                for result in validation_results.values()
                if isinstance(result, dict)
            )
            
            return validation_results
            
        except Exception as e:
            logger.error("Error validating against ground truth: %s", str(e))
            return {"valid": False, "error": str(e)}
    
    def set_ground_truth_data(self, session_id: str, ground_truth: QualityMetrics):
        """Set ground truth data for a session"""
        self.ground_truth_data[session_id] = ground_truth
        logger.info("Ground truth data set for session %s", session_id)
    
    def get_quality_report(self, session_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive quality report for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict[str, Any]: Quality report
        """
        if session_id not in self.metrics_history:
            return {"error": "No metrics available for session"}
        
        metrics_list = self.metrics_history[session_id]
        if not metrics_list:
            return {"error": "No metrics available for session"}
        
        latest_metrics = metrics_list[-1]
        statistical_summary = self.calculate_statistical_summary(session_id)
        trend_analysis = {}
        
        for metric_type in MetricType:
            trend_analysis[metric_type.value] = self.detect_metric_trends(session_id, metric_type)
        
        # Check against thresholds
        threshold_checks = self._check_metric_thresholds(latest_metrics)
        
        # Convert to dict for JSON serialization
        if DATACLASSES_JSON_AVAILABLE:
            latest_metrics_dict = latest_metrics.to_dict()
        else:
            latest_metrics_dict = {
                "session_id": latest_metrics.session_id,
                "timestamp": latest_metrics.timestamp,
                "wer": latest_metrics.wer,
                "der": latest_metrics.der,
                "si_sdr": latest_metrics.si_sdr,
                "precision": latest_metrics.precision,
                "recall": latest_metrics.recall,
                "f1_score": latest_metrics.f1_score,
                "confidence": latest_metrics.confidence,
                "metadata": latest_metrics.metadata
            }
        
        return {
            "session_id": session_id,
            "timestamp": time.time(),
            "latest_metrics": latest_metrics_dict,
            "statistical_summary": {
                mtype.value: summary.__dict__ 
                for mtype, summary in statistical_summary.items()
            },
            "trend_analysis": trend_analysis,
            "threshold_checks": threshold_checks,
            "metrics_history_count": len(metrics_list)
        }
    
    def _check_metric_thresholds(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """Check metrics against quality thresholds"""
        checks = {}
        
        # WER check
        if metrics.wer is not None:
            if metrics.wer <= self.thresholds.wer_warning:
                checks["wer"] = {"status": "good", "value": metrics.wer}
            elif metrics.wer <= self.thresholds.wer_critical:
                checks["wer"] = {"status": "warning", "value": metrics.wer}
            else:
                checks["wer"] = {"status": "critical", "value": metrics.wer}
        
        # DER check
        if metrics.der is not None:
            if metrics.der <= self.thresholds.der_warning:
                checks["der"] = {"status": "good", "value": metrics.der}
            elif metrics.der <= self.thresholds.der_critical:
                checks["der"] = {"status": "warning", "value": metrics.der}
            else:
                checks["der"] = {"status": "critical", "value": metrics.der}
        
        # SI-SDR check
        if metrics.si_sdr is not None:
            if metrics.si_sdr >= self.thresholds.si_sdr_warning:
                checks["si_sdr"] = {"status": "good", "value": metrics.si_sdr}
            elif metrics.si_sdr >= self.thresholds.si_sdr_critical:
                checks["si_sdr"] = {"status": "warning", "value": metrics.si_sdr}
            else:
                checks["si_sdr"] = {"status": "critical", "value": metrics.si_sdr}
        
        # Precision check
        if metrics.precision is not None:
            if metrics.precision >= self.thresholds.precision_warning:
                checks["precision"] = {"status": "good", "value": metrics.precision}
            elif metrics.precision >= self.thresholds.precision_critical:
                checks["precision"] = {"status": "warning", "value": metrics.precision}
            else:
                checks["precision"] = {"status": "critical", "value": metrics.precision}
        
        return checks

# Factory function for easy creation
def create_metrics_calculator(config: Optional[Dict[str, Any]] = None) -> MetricsCalculator:
    """
    Create a MetricsCalculator instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MetricsCalculator: Metrics calculator instance
    """
    return MetricsCalculator(config)

# Example usage and testing
def example_usage():
    """Example demonstrating metrics calculator usage"""
    
    # Create metrics calculator
    calculator = create_metrics_calculator({
        "sample_rate": 16000,
        "wer_collar": 0.25
    })
    
    # Example WER calculation
    reference = "the quick brown fox jumps over the lazy dog"
    hypothesis = "the quick brown fox jumped over the lazy dog"
    
    wer, alignment = calculator.calculate_wer(reference, hypothesis)
    print(f"WER: {wer:.4f}")
    print(f"Alignment: {alignment.alignment[:5]}...")  # Show first 5 alignments
    
    # Example DER calculation
    ref_segments = [
        DiarizationSegment(0.0, 2.0, "speaker_1"),
        DiarizationSegment(2.0, 4.0, "speaker_2")
    ]
    hyp_segments = [
        DiarizationSegment(0.1, 1.9, "speaker_1"),
        DiarizationSegment(2.1, 3.9, "speaker_2")
    ]
    
    der = calculator.calculate_der(ref_segments, hyp_segments)
    print(f"DER: {der:.4f}")
    
    # Example SI-SDR calculation
    duration = 1.0  # 1 second
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    reference_audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    enhanced_audio = reference_audio + 0.1 * np.random.randn(len(reference_audio))  # Add noise
    
    si_sdr = calculator.calculate_si_sdr(reference_audio, enhanced_audio, sample_rate)
    print(f"SI-SDR: {si_sdr:.2f} dB")
    
    # Example speaker metrics
    ref_labels = ["speaker_1", "speaker_1", "speaker_2", "speaker_2"]
    hyp_labels = ["speaker_1", "speaker_2", "speaker_2", "speaker_1"]  # Some errors
    confidence = [0.9, 0.8, 0.95, 0.7]
    
    speaker_metrics = calculator.calculate_speaker_metrics(ref_labels, hyp_labels, confidence)
    print(f"Speaker Precision: {speaker_metrics['precision']:.4f}")
    print(f"Speaker Recall: {speaker_metrics['recall']:.4f}")
    print(f"Speaker F1-Score: {speaker_metrics['f1_score']:.4f}")

if __name__ == "__main__":
    example_usage()
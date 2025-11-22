"""
DeepSight AI Vision REST API Service
"""

import json
import os
import base64
import logging
import cv2
import numpy as np
import re
import random
import hashlib
import threading
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from ultralytics import YOLO
from paddleocr import PaddleOCR
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz
from cachetools import TTLCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global AI model instances
yolo_model = None
paddle_ocr = None


# ============================================================================
# Semantic Matching Configuration
# ============================================================================

class SemanticConfig:
    """Configuration for semantic matching from environment variables"""
    
    def __init__(self):
        self.enable_semantic_match = os.getenv("ENABLE_SEMANTIC_MATCH", "false").lower() == "true"
        self.semantic_rollout_percent = int(os.getenv("SEMANTIC_ROLLOUT_PERCENT", "100"))
        self.allow_client_override = os.getenv("ALLOW_CLIENT_OVERRIDE", "true").lower() == "true"
        self.semantic_threshold = float(os.getenv("SEMANTIC_THRESHOLD", "90.0"))
        self.semantic_alpha = float(os.getenv("SEMANTIC_ALPHA", "0.7"))
        
        # Default model (English, lightweight)
        self.semantic_model_name = os.getenv("SEMANTIC_MODEL_NAME", "all-MiniLM-L6-v2")
        
        # Multilingual model (optional, feature-flagged)
        self.enable_multilingual_model = os.getenv("ENABLE_MULTILINGUAL_MODEL", "false").lower() == "true"
        self.multilingual_model_name = os.getenv("MULTILINGUAL_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
        
        self.cache_max_size = int(os.getenv("SEMANTIC_CACHE_MAX_SIZE", "10000"))
        self.cache_ttl_seconds = int(os.getenv("SEMANTIC_CACHE_TTL", "3600"))
        
        logger.info(f"Semantic Config: enabled={self.enable_semantic_match}, "
                   f"rollout={self.semantic_rollout_percent}%, "
                   f"threshold={self.semantic_threshold}, "
                   f"alpha={self.semantic_alpha}, "
                   f"default_model={self.semantic_model_name}, "
                   f"multilingual_enabled={self.enable_multilingual_model}, "
                   f"multilingual_model={self.multilingual_model_name}, "
                   f"cache_max={self.cache_max_size}, "
                   f"cache_ttl={self.cache_ttl_seconds}s")


# Global semantic configuration
semantic_config = SemanticConfig()


# ============================================================================
# Semantic Matcher with Hybrid Scoring
# ============================================================================

class SemanticMatcher:
    """
    Hybrid semantic + fuzzy text matcher with dual-model support
    Supports both default (English) and multilingual models
    """
    
    def __init__(
        self, 
        default_model_name: str = "all-MiniLM-L6-v2",
        multilingual_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        cache_max_size: int = 10000, 
        cache_ttl: int = 3600
    ):
        """
        Initialize semantic matcher with dual model support
        
        Args:
            default_model_name: Default English model
            multilingual_model_name: Optional multilingual model
            cache_max_size: Maximum number of embeddings to cache
            cache_ttl: Time-to-live for cached embeddings in seconds
        """
        # Model names
        self.default_model_name = default_model_name
        self.multilingual_model_name = multilingual_model_name
        
        # Models (lazy loaded)
        self.default_model = None
        self.multilingual_model = None
        self.model_lock = threading.Lock()  # Thread-safe model loading
        
        # Separate caches for each model
        self.default_cache = TTLCache(maxsize=cache_max_size, ttl=cache_ttl)
        self.multilingual_cache = TTLCache(maxsize=cache_max_size, ttl=cache_ttl)
        self.cache_lock = threading.Lock()  # Thread-safe cache access
        
        # Stats per model
        self.default_cache_hits = 0
        self.default_cache_misses = 0
        self.multilingual_cache_hits = 0
        self.multilingual_cache_misses = 0
        self.stats_lock = threading.Lock()  # Thread-safe counter updates
        
        logger.info(f"SemanticMatcher initialized: "
                   f"default_model={default_model_name}, "
                   f"multilingual_model={multilingual_model_name}, "
                   f"cache_max={cache_max_size}, ttl={cache_ttl}s")
    
    def _load_model(self, use_multilingual: bool = False):
        """
        Lazy load the sentence transformer model on CPU (thread-safe)
        
        Args:
            use_multilingual: If True, load multilingual model; else load default model
        """
        model_attr = "multilingual_model" if use_multilingual else "default_model"
        model_name = self.multilingual_model_name if use_multilingual else self.default_model_name
        
        if getattr(self, model_attr) is None:
            with self.model_lock:
                # Double-check locking pattern
                if getattr(self, model_attr) is None:
                    logger.info(f"Loading {'multilingual' if use_multilingual else 'default'} "
                               f"sentence transformer model: {model_name}")
                    setattr(self, model_attr, SentenceTransformer(model_name, device="cpu"))
                    logger.info(f"Model {model_name} loaded successfully on CPU")
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching"""
        if not text:
            return ""
        # Trim, collapse whitespace
        normalized = " ".join(text.strip().split())
        return normalized
    
    def _get_cache_key(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate cache key for embeddings with better context isolation
        
        Args:
            text: Text to encode
            context: Optional context dict with tenant_id, page_hash, etc.
        
        Returns:
            Cache key (MD5 hash)
        """
        if context:
            tenant = context.get("tenant_id", "default")
            page = context.get("page_hash", "default")
            key = f"{tenant}:{page}:{text}"
        else:
            key = f"default:default:{text}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def get_cache_stats(self, use_multilingual: bool = False) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring (thread-safe)
        
        Args:
            use_multilingual: Get stats for multilingual model if True, else default model
        """
        with self.stats_lock:
            if use_multilingual:
                cache = self.multilingual_cache
                hits = self.multilingual_cache_hits
                misses = self.multilingual_cache_misses
                model = "multilingual"
            else:
                cache = self.default_cache
                hits = self.default_cache_hits
                misses = self.default_cache_misses
                model = "default"
            
            return {
                "model": model,
                "cache_size": len(cache),
                "cache_max_size": cache.maxsize,
                "cache_hits": hits,
                "cache_misses": misses,
                "hit_rate": round(hits / max(hits + misses, 1) * 100, 2)
            }
    
    def _encode_with_cache(
        self, 
        texts: List[str], 
        context: Optional[Dict[str, Any]] = None,
        use_multilingual: bool = False
    ) -> np.ndarray:
        """
        Encode texts with TTL-based caching
        
        Args:
            texts: List of texts to encode
            context: Optional context dict for cache partitioning (tenant_id, page_hash, etc.)
            use_multilingual: If True, use multilingual model; else use default model
            
        Returns:
            Numpy array of embeddings
        """
        self._load_model(use_multilingual)
        
        # Select appropriate model and cache
        model = self.multilingual_model if use_multilingual else self.default_model
        cache = self.multilingual_cache if use_multilingual else self.default_cache
        
        # First pass: identify cached vs uncached texts (thread-safe)
        texts_to_encode = []
        texts_to_encode_indices = []
        
        with self.cache_lock:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text, context)
                if cache_key in cache:
                    with self.stats_lock:
                        if use_multilingual:
                            self.multilingual_cache_hits += 1
                        else:
                            self.default_cache_hits += 1
                else:
                    with self.stats_lock:
                        if use_multilingual:
                            self.multilingual_cache_misses += 1
                        else:
                            self.default_cache_misses += 1
                    texts_to_encode.append(text)
                    texts_to_encode_indices.append(i)
        
        # Batch encode uncached texts (outside lock - can be slow)
        if texts_to_encode:
            new_embeddings = model.encode(texts_to_encode, convert_to_tensor=False, show_progress_bar=False)
            
            # Cache new embeddings (thread-safe)
            with self.cache_lock:
                for idx, text in enumerate(texts_to_encode):
                    cache_key = self._get_cache_key(text, context)
                    cache[cache_key] = new_embeddings[idx]
        
        # Second pass: build embeddings list in correct order (thread-safe)
        embeddings = []
        with self.cache_lock:
            for text in texts:
                cache_key = self._get_cache_key(text, context)
                embeddings.append(cache[cache_key])
        
        # Log cache stats periodically
        if use_multilingual:
            total_requests = self.multilingual_cache_hits + self.multilingual_cache_misses
        else:
            total_requests = self.default_cache_hits + self.default_cache_misses
            
        if total_requests % 100 == 0:
            stats = self.get_cache_stats(use_multilingual)
            logger.info(f"Cache stats: {stats}")
        
        return np.array(embeddings)
    
    def hybrid_match(
        self,
        query_text: str,
        candidates: List[Dict[str, Any]],
        alpha: float = 0.7,
        threshold: float = 90.0,
        context: Optional[Dict[str, Any]] = None,
        use_multilingual: bool = False
    ) -> Dict[str, Any]:
        """
        Perform hybrid semantic + fuzzy matching
        
        Args:
            query_text: Text to search for
            candidates: List of candidate elements with 'text' field
            alpha: Weight for semantic score (0-1), fuzzy weight = 1-alpha
            threshold: Minimum combined score threshold (0-100)
            context: Optional context dict (tenant_id, page_hash, viewport_id, etc.)
            use_multilingual: If True, use multilingual model; else use default model
            
        Returns:
            Dictionary with matching results
        """
        if not candidates:
            return {
                "matched": False,
                "reason": "no_candidates",
                "best_candidate": None,
                "top_candidates": []
            }
        
        # Normalize query
        query_normalized = self._normalize_text(query_text)
        query_lower = query_normalized.lower()
        
        # Prepare candidate texts
        candidate_texts = []
        candidate_indices = []
        
        for idx, candidate in enumerate(candidates):
            text = candidate.get("text", "")
            if text:
                candidate_texts.append(self._normalize_text(text))
                candidate_indices.append(idx)
        
        if not candidate_texts:
            return {
                "matched": False,
                "reason": "no_valid_candidate_texts",
                "best_candidate": None,
                "top_candidates": []
            }
        
        # Encode query and candidates with selected model
        query_embedding = self._encode_with_cache([query_normalized], context, use_multilingual)[0]
        candidate_embeddings = self._encode_with_cache(candidate_texts, context, use_multilingual)
        
        # Compute semantic similarity
        cosine_scores = util.cos_sim(query_embedding, candidate_embeddings)[0].cpu().numpy()
        
        # Compute fuzzy scores and combine
        results = []
        for i, candidate_idx in enumerate(candidate_indices):
            candidate = candidates[candidate_idx]
            candidate_text = candidate_texts[i]
            
            # Semantic score: convert cosine [-1, 1] to percentage [0, 100]
            cosine_sim = float(cosine_scores[i])
            semantic_pct = ((cosine_sim + 1) / 2) * 100
            
            # Fuzzy score
            fuzzy_pct = fuzz.ratio(query_lower, candidate_text.lower(), score_cutoff=0)
            
            # Combined score
            combined_pct = alpha * semantic_pct + (1 - alpha) * fuzzy_pct
            
            results.append({
                "candidate": candidate,
                "candidate_text": candidate_text,
                "semantic_pct": round(semantic_pct, 2),
                "fuzzy_pct": round(fuzzy_pct, 2),
                "combined_pct": round(combined_pct, 2),
                "cosine_similarity": round(cosine_sim, 4)
            })
        
        # Sort by combined score
        results.sort(key=lambda x: x["combined_pct"], reverse=True)
        
        # Get top 5
        top_5 = results[:5]
        best = results[0] if results else None
        
        model_name = self.multilingual_model_name if use_multilingual else self.default_model_name
        
        if best and best["combined_pct"] >= threshold:
            return {
                "matched": True,
                "best_candidate": best,
                "top_candidates": top_5,
                "model": model_name,
                "use_multilingual": use_multilingual,
                "alpha": alpha,
                "threshold": threshold
            }
        else:
            return {
                "matched": False,
                "reason": "below_threshold",
                "best_candidate": best,
                "top_candidates": top_5,
                "model": model_name,
                "use_multilingual": use_multilingual,
                "alpha": alpha,
                "threshold": threshold
            }


# Global semantic matcher instance
semantic_matcher = None


# ============================================================================
# Lifespan Event Handler
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    global yolo_model, paddle_ocr, semantic_matcher
    
    # Startup
    try:
        logger.info("Starting DeepSight AI Vision API...")
        
        # Get model path
        model_path = os.getenv(
            "YOLO_MODEL_PATH",
            os.path.realpath(
                os.path.join(os.path.dirname(__file__), 'trained_models/sample_trained_model.pt')
            )
        )
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Initialize DeepSight AI Vision Core
        confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.05"))
        ai_core = AIVisionCore(model_path, confidence_threshold)
        
        yolo_model = ai_core.model
        paddle_ocr = ai_core.paddle_ocr
        
        # Initialize Semantic Matcher with dual model support (lazy loading)
        semantic_matcher = SemanticMatcher(
            default_model_name=semantic_config.semantic_model_name,
            multilingual_model_name=semantic_config.multilingual_model_name,
            cache_max_size=semantic_config.cache_max_size,
            cache_ttl=semantic_config.cache_ttl_seconds
        )
        
        # Store in app state
        app.state.ai_core = ai_core
        app.state.semantic_matcher = semantic_matcher
        app.state.semantic_config = semantic_config
        
        logger.info("DeepSight AI Vision API started successfully")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Confidence threshold: {confidence_threshold}")
        
    except Exception as e:
        logger.error(f"Failed to initialize AI models: {str(e)}")
        raise
    
    yield  # Application is running
    
    # Shutdown
    logger.info("DeepSight AI Vision API shutting down")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="DeepSight Vision API",
    description="General Purpose AI Computer Vision API for UI element detection",
    version="1.0.0",
    lifespan=lifespan
)



# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class ViewportSize(BaseModel):
    width: int
    height: int


class AnalyzePageRequest(BaseModel):
    screenshot_base64: str = Field(..., description="Base64 encoded screenshot image")
    viewport_size: ViewportSize = Field(..., description="Browser viewport dimensions")
    confidence_threshold: Optional[float] = Field(0.05, description="YOLO confidence threshold")
    
    # New fields for semantic matching
    text: Optional[str] = Field(None, description="Expected UI text to locate (triggers semantic matching if provided)")
    element_type: Optional[str] = Field(None, description="Type of element (e.g., 'floating_label')")
    
    # Per-request overrides for semantic matching
    enable_semantic_match: Optional[bool] = Field(None, description="Override server-level semantic matching enable flag")
    semantic_threshold: Optional[float] = Field(None, description="Semantic match threshold percentage (0-100)")
    semantic_alpha: Optional[float] = Field(None, description="Alpha weight for semantic score (0-1)")
    
    # Multilingual model selection (feature flag)
    use_multilingual_model: Optional[bool] = Field(None, description="Use multilingual model instead of default English model")
    
    # Telemetry fields
    tenant_id: Optional[str] = Field(None, description="Tenant identifier for telemetry")
    request_id: Optional[str] = Field(None, description="Request identifier for telemetry and rollout")


class ElementSearchCriteria(BaseModel):
    text: Optional[str] = Field(None, description="Text to search for")
    class_name: Optional[str] = Field(None, description="Element class name (e.g., 'field', 'button')")
    element_type: Optional[str] = Field(None, description="Element type (e.g., 'floating_label', 'non-text')")


class FindElementRequest(BaseModel):
    screenshot_base64: str = Field(..., description="Base64 encoded screenshot image")
    viewport_size: ViewportSize = Field(..., description="Browser viewport dimensions")
    search_criteria: ElementSearchCriteria = Field(..., description="Search criteria for element")
    confidence_threshold: Optional[float] = Field(0.05, description="YOLO confidence threshold")


class FilterElementsRequest(BaseModel):
    screenshot_base64: str = Field(..., description="Base64 encoded screenshot image")
    viewport_size: ViewportSize = Field(..., description="Browser viewport dimensions")
    text: Optional[str] = Field(None, description="Filter by text content")
    element_type: Optional[str] = Field(None, description="Filter by element type")
    class_name: Optional[str] = Field(None, description="Filter by class name")
    confidence_threshold: Optional[float] = Field(0.05, description="YOLO confidence threshold")


class DetectedElement(BaseModel):
    type: str
    text: Optional[str] = None
    class_name: Optional[str] = None
    x: int
    y: int
    width: int
    height: int
    x1: int
    y1: int
    x2: int
    y2: int
    scaled_center_x: int
    scaled_center_y: int
    confidence: Optional[float] = None


class AnalyzePageResponse(BaseModel):
    screenshot_size: Dict[str, int]
    detected_elements: List[Dict[str, Any]]
    total_text_elements: int
    total_non_text_elements: int
    
    # Semantic matching results (optional, present if text search was requested)
    matched: Optional[bool] = None
    method: Optional[str] = None  # "direct" or "semantic"
    reason: Optional[str] = None  # Reason if not matched
    semantic_result: Optional[Dict[str, Any]] = None  # Detailed semantic matching result


class FindElementResponse(BaseModel):
    found: bool
    element: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    yolo_model_loaded: bool
    paddle_ocr_loaded: bool
    model_path: Optional[str] = None


# ============================================================================
# DeepSight AI Vision Core Logic
# ============================================================================

class AIVisionCore:
    """Core DeepSight AI Vision logic - no Playwright dependencies"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.05):
        """
        Initialize DeepSight AI Vision Core
        
        Args:
            model_path: Path to YOLO model file
            confidence_threshold: Confidence threshold for YOLO detection
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(model_path)
        self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")
        logger.info(f"DeepSight AI Vision Core initialized with model: {model_path}")
    
    def decode_base64_image(self, base64_string: str) -> np.ndarray:
        """
        Decode base64 string to numpy array image
        
        Args:
            base64_string: Base64 encoded image
            
        Returns:
            numpy array image
        """
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_string)
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # Decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Failed to decode image")
            
            return img
        except Exception as e:
            logger.error(f"Error decoding base64 image: {str(e)}")
            raise
    
    def detect_ui_elements(
        self,
        img: np.ndarray,
        viewport_width: int,
        viewport_height: int
    ) -> List[Dict[str, Any]]:
        """
        Detect both text (PaddleOCR) and non-text (YOLO) UI elements
        
        Args:
            img: OpenCV image (numpy array)
            viewport_width: Browser viewport width (equivalent to screen_width in original)
            viewport_height: Browser viewport height (equivalent to screen_height in original)
            
        Returns:
            List of detected elements with coordinates and metadata
        """
        detected_elements = []
        
        original_height, original_width = img.shape[:2]
        
        # Define delimiters for text splitting
        delimiters = ["...", "?"]  # Add more delimiters as needed
        
        # Add regex pattern support for delimiters
        regex_delimiters = [r'O\s*', r"(?<=\s)O(?=\s)"]  # Regex for 'O' followed by spaces (radio button)
        
        # Process text elements with PaddleOCR
        results = self.paddle_ocr.ocr(img, cls=True)
        
        # Pre-calculate screen dimensions (passed as parameters in API version)
        screen_width, screen_height = viewport_width, viewport_height
        
        for line in results:
            for word in line:
                bbox, (text, confidence) = word
                x1, y1 = map(int, bbox[0])
                x2, y2 = map(int, bbox[2])
                width = x2 - x1
                height = y2 - y1
                
                # Check for delimiter match once
                split_info = None
                for delimiter in delimiters:
                    if delimiter in text:
                        split_index = text.index(delimiter) + len(delimiter)
                        if split_index < len(text):
                            split_info = {
                                'delimiter': delimiter,
                                'index': split_index,
                                'pre_text': text[:split_index].strip(),
                                'post_text': text[split_index:].strip()
                            }
                            break
                
                # Check for regex delimiter match if no literal delimiter was found
                if not split_info:
                    for pattern in regex_delimiters:
                        match = re.search(pattern, text)
                        if match:
                            split_index = match.end()
                            if split_index < len(text):
                                split_info = {
                                    'delimiter': match.group(),
                                    'index': split_index,
                                    'pre_text': text[:split_index].strip(),
                                    'post_text': text[split_index:].strip()
                                }
                                break
                
                if split_info and split_info['pre_text'] and split_info['post_text']:
                    # Calculate dimensions once
                    avg_char_width = width / len(text)
                    spacing_width = int(avg_char_width)
                    pre_width = int((len(split_info['pre_text']) / len(text)) * width)
                    post_width = width - pre_width - spacing_width
                    
                    # Calculate centers once
                    first_center_x = x1 + (pre_width // 2)
                    second_center_x = x1 + pre_width + spacing_width + (post_width // 2)
                    center_y = (y1 + y2) // 2
                    
                    # Scale coordinates in batch
                    scaled_first_x = int((first_center_x / original_width) * screen_width)
                    scaled_second_x = int((second_center_x / original_width) * screen_width)
                    scaled_y = int((center_y / original_height) * screen_height)
                    
                    # First part
                    detected_elements.append({
                        "type": "floating_label",
                        "text": split_info['pre_text'],
                        "x": x1,
                        "y": y1,
                        "width": pre_width,
                        "height": height,
                        "confidence": confidence * 100,
                        "x1": x1,
                        "x2": x1 + pre_width,
                        "y1": y1,
                        "y2": y2,
                        "scaled_center_x": scaled_first_x,
                        "scaled_center_y": scaled_y
                    })
                    
                    # Second part
                    second_x = x1 + pre_width + spacing_width
                    detected_elements.append({
                        "type": "floating_label",
                        "text": split_info['post_text'],
                        "x": second_x,
                        "y": y1,
                        "width": post_width,
                        "height": height,
                        "confidence": confidence * 100,
                        "x1": second_x,
                        "x2": second_x + post_width,
                        "y1": y1,
                        "y2": y2,
                        "scaled_center_x": scaled_second_x,
                        "scaled_center_y": scaled_y
                    })
                else:
                    # Regular text element - calculate scaled coordinates directly
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    scaled_center_x = int((center_x / original_width) * screen_width)
                    scaled_center_y = int((center_y / original_height) * screen_height)
                    
                    detected_elements.append({
                        "type": "floating_label",
                        "text": text,
                        "x": x1,
                        "y": y1,
                        "width": width,
                        "height": height,
                        "confidence": confidence * 100,
                        "x1": x1,
                        "x2": x2,
                        "y1": y1,
                        "y2": y2,
                        "scaled_center_x": scaled_center_x,
                        "scaled_center_y": scaled_center_y
                    })
        
        logger.info(f"[OCR] Floating labels detected: {len([e for e in detected_elements if e['type'] == 'floating_label'])}")
        
        # Process non-text UI elements with YOLO
        # Pass numpy array directly to YOLO
        results = self.model.predict(source=img, conf=self.confidence_threshold)
        
        for box in results[0].boxes:
            class_name = results[0].names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width = x2 - x1
            height = y2 - y1
            
            # Calculate scaled coordinates directly
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            scaled_center_x = int((center_x / original_width) * screen_width)
            scaled_center_y = int((center_y / original_height) * screen_height)
            
            detected_elements.append({
                "type": "non-text",
                "class_name": class_name,
                "x": x1,
                "y": y1,
                "width": width,
                "height": height,
                "x1": x1,
                "x2": x2,
                "y1": y1,
                "y2": y2,
                "scaled_center_x": scaled_center_x,
                "scaled_center_y": scaled_center_y
            })
        
        logger.info(
            f"[YOLO] Non-text elements detected: {len([e for e in detected_elements if e['type'] == 'non-text'])}")
        
        return detected_elements
    
    def find_element(
        self,
        detected_elements: List[Dict[str, Any]],
        search_criteria: ElementSearchCriteria
    ) -> Optional[Dict[str, Any]]:
        """
        Find element in detected elements based on search criteria
        
        Args:
            detected_elements: List of detected elements
            search_criteria: Search criteria
            
        Returns:
            Found element or None
        """
        for element in detected_elements:
            # Build match criteria - ALL specified criteria must match
            matches = True
            
            # Check text match
            if search_criteria.text:
                element_text = element.get("text", "")
                if element_text.lower().strip() != search_criteria.text.lower().strip():
                    matches = False
            
            # Check class name match
            if search_criteria.class_name:
                if element.get("class_name") != search_criteria.class_name:
                    matches = False
            
            # Check type match
            if search_criteria.element_type:
                if element.get("type") != search_criteria.element_type:
                    matches = False
            
            # Return element only if ALL criteria match
            if matches:
                return element
        
        return None
    
    def filter_elements(
        self,
        detected_elements: List[Dict[str, Any]],
        text: Optional[str] = None,
        element_type: Optional[str] = None,
        class_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter detected elements by text, type, or class name
        
        Args:
            detected_elements: List of detected elements
            text: Filter by text content (case-insensitive)
            element_type: Filter by element type
            class_name: Filter by class name
            
        Returns:
            Filtered list of elements
        """
        filtered = detected_elements
        
        if text:
            search_text = text.lower().strip()
            filtered = [
                e for e in filtered 
                if e.get("text", "").lower().strip() == search_text
            ]
        
        if element_type:
            filtered = [e for e in filtered if e.get("type") == element_type]
        
        if class_name:
            filtered = [e for e in filtered if e.get("class_name") == class_name]
        
        return filtered


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "name": "DeepSight AI Vision API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Returns service status and model availability
    """
    return HealthResponse(
        status="healthy" if yolo_model and paddle_ocr else "unhealthy",
        yolo_model_loaded=yolo_model is not None,
        paddle_ocr_loaded=paddle_ocr is not None,
        model_path=app.state.ai_core.model_path if hasattr(app.state, 'ai_core') else None
    )


@app.post("/analyze/page", response_model=AnalyzePageResponse)
async def analyze_page(request: AnalyzePageRequest):
    """
    Analyze page screenshot and return complete spatial mapping with optional semantic text matching
    
    This endpoint processes a screenshot and returns all detected UI elements.
    If 'text' and 'element_type' are provided, performs direct lookup followed by semantic fallback.
    """
    try:
        ai_core: AIVisionCore = app.state.ai_core
        semantic_matcher_inst: SemanticMatcher = app.state.semantic_matcher
        config: SemanticConfig = app.state.semantic_config
        
        # Update confidence threshold if provided
        if request.confidence_threshold:
            ai_core.confidence_threshold = request.confidence_threshold
        
        # Decode base64 image (validates image format)
        img = ai_core.decode_base64_image(request.screenshot_base64)
        
        # Detect UI elements (YOLO + PaddleOCR)
        detected_elements = ai_core.detect_ui_elements(
            img,
            request.viewport_size.width,
            request.viewport_size.height
        )
            
        # Prepare base response
        text_elements = [e for e in detected_elements if e["type"] == "floating_label"]
        non_text_elements = [e for e in detected_elements if e["type"] == "non-text"]
        
        screenshot_size = {
            "width": img.shape[1],
            "height": img.shape[0]
        }
        
        # Check if semantic matching is requested
        if request.text and request.element_type:
            logger.info(f"Text search requested: text='{request.text}', type='{request.element_type}'")
            
            # Step B: Direct lookup with normalization
            matched_element, method, reason = _perform_direct_lookup(
                request.text, 
                request.element_type, 
                detected_elements
            )
            
            if matched_element:
                # Direct match found
                logger.info(f"Direct match found for text='{request.text}'")
                return AnalyzePageResponse(
                    screenshot_size=screenshot_size,
                    detected_elements=detected_elements,
                    total_text_elements=len(text_elements),
                    total_non_text_elements=len(non_text_elements),
                    matched=True,
                    method="direct",
                    reason=None
                )
            
            # Step C: No direct match - check if semantic fallback is allowed
            semantic_enabled = _is_semantic_enabled(config, request)
            
            if not semantic_enabled:
                logger.info(f"Semantic fallback disabled for text='{request.text}'")
                return AnalyzePageResponse(
                    screenshot_size=screenshot_size,
                    detected_elements=detected_elements,
                    total_text_elements=len(text_elements),
                    total_non_text_elements=len(non_text_elements),
                    matched=False,
                    method="direct",
                    reason="semantic_disabled"
                )
                
            # Semantic fallback is allowed - perform hybrid matching
            logger.info(f"Performing semantic fallback for text='{request.text}'")
            
            # Get parameters (with overrides)
            alpha = request.semantic_alpha if request.semantic_alpha is not None else config.semantic_alpha
            threshold = request.semantic_threshold if request.semantic_threshold is not None else config.semantic_threshold
            
            # Determine which model to use (feature flag)
            use_multilingual = False
            if request.use_multilingual_model is not None:
                # Per-request override
                use_multilingual = request.use_multilingual_model
            elif config.enable_multilingual_model:
                # Server-level default
                use_multilingual = True
            
            model_type = "multilingual" if use_multilingual else "default"
            logger.info(f"Using {model_type} model for semantic matching")
            
            # Filter candidates by element_type
            candidates = [e for e in detected_elements if e.get("type") == request.element_type]
            
            if not candidates:
                logger.warning(f"No candidates found for element_type='{request.element_type}'")
                return AnalyzePageResponse(
                    screenshot_size=screenshot_size,
                    detected_elements=detected_elements,
                    total_text_elements=len(text_elements),
                    total_non_text_elements=len(non_text_elements),
                    matched=False,
                    method="semantic",
                    reason="no_candidates_for_type"
                )
                
            # Perform hybrid matching with context for better cache isolation
            cache_context = {
                "tenant_id": request.tenant_id or "default",
                "viewport_id": request.request_id or f"{request.viewport_size.width}x{request.viewport_size.height}",
                "page_hash": hashlib.md5(request.screenshot_base64[:100].encode()).hexdigest()[:8] if request.screenshot_base64 else "default"
            }
            
            match_result = semantic_matcher_inst.hybrid_match(
                query_text=request.text,
                candidates=candidates,
                alpha=alpha,
                threshold=threshold,
                context=cache_context,
                use_multilingual=use_multilingual
            )
                
            if match_result["matched"]:
                # Semantic match found - create semantic elements for ALL instances of the best match
                best = match_result["best_candidate"]
                all_matches = match_result["top_candidates"]
                
                # Get the text of the best matching candidate
                best_candidate_text = best["candidate_text"].strip().lower()
                
                # Filter to return ONLY candidates with the SAME text as best match
                # This ensures we get all instances of "Login" but not "Password", "Username", etc.
                matched_candidates = [
                    candidate for candidate in all_matches 
                    if candidate["candidate_text"].strip().lower() == best_candidate_text
                ]
                
                logger.info(f"Found {len(matched_candidates)} instances of '{best['candidate_text']}' "
                           f"(ignoring other texts even if above threshold)")
                
                # Enrich existing elements with semantic metadata (avoid duplicates!)
                semantic_elements_enriched = []
                for match in matched_candidates:
                    matched_candidate = match["candidate"]
                    
                    # Find the existing element in detected_elements
                    # Match by text, type, and coordinates
                    existing_element = None
                    for elem in detected_elements:
                        if (elem.get("text") == matched_candidate.get("text") and
                            elem.get("type") == request.element_type and
                            elem.get("x") == matched_candidate.get("x") and
                            elem.get("y") == matched_candidate.get("y")):
                            existing_element = elem
                            break
                    
                    if existing_element:
                        # Enrich the existing element with semantic metadata
                        existing_element["text_source"] = "semantic_query"
                        existing_element["semantic_query"] = request.text
                        existing_element["semantic_pct"] = match["semantic_pct"]
                        existing_element["fuzzy_pct"] = match["fuzzy_pct"]
                        existing_element["combined_pct"] = match["combined_pct"]
                        existing_element["model"] = match_result["model"]
                        existing_element["method"] = "semantic"
                        
                        semantic_elements_enriched.append(existing_element)
                        
                        logger.info(f"Semantic match: query='{request.text}' -> found='{match['candidate_text']}' "
                                   f"at ({matched_candidate.get('x')}, {matched_candidate.get('y')}) "
                                   f"(combined_pct={match['combined_pct']}) [enriched existing element]")
                    else:
                        # Element not found (shouldn't happen, but create new as fallback)
                        logger.warning(f"Semantic match candidate not found in detected_elements, creating new element")
                        
                        semantic_element = {
                            "text": matched_candidate.get("text"),
                            "text_source": "semantic_query",
                            "semantic_query": request.text,
                            "type": request.element_type,
                            "x": matched_candidate.get("x"),
                            "y": matched_candidate.get("y"),
                            "width": matched_candidate.get("width"),
                            "height": matched_candidate.get("height"),
                            "x1": matched_candidate.get("x1"),
                            "x2": matched_candidate.get("x2"),
                            "y1": matched_candidate.get("y1"),
                            "y2": matched_candidate.get("y2"),
                            "scaled_center_x": matched_candidate.get("scaled_center_x"),
                            "scaled_center_y": matched_candidate.get("scaled_center_y"),
                            "confidence": matched_candidate.get("confidence"),
                            "semantic_pct": match["semantic_pct"],
                            "fuzzy_pct": match["fuzzy_pct"],
                            "combined_pct": match["combined_pct"],
                            "model": match_result["model"],
                            "method": "semantic"
                        }
                        
                        detected_elements.append(semantic_element)
                        semantic_elements_enriched.append(semantic_element)
                        
                        if request.element_type == "floating_label":
                            text_elements.append(semantic_element)
                        
                        logger.info(f"Semantic match: query='{request.text}' -> found='{match['candidate_text']}' "
                                   f"at ({matched_candidate.get('x')}, {matched_candidate.get('y')}) "
                                   f"(combined_pct={match['combined_pct']}) [created new element]")
                
                return AnalyzePageResponse(
                    screenshot_size=screenshot_size,
                    detected_elements=detected_elements,
                    total_text_elements=len(text_elements),
                    total_non_text_elements=len(non_text_elements),
                    matched=True,
                    method="semantic",
                    semantic_result={
                        "best_candidate": {
                            "text": best["candidate_text"],
                            "semantic_pct": best["semantic_pct"],
                            "fuzzy_pct": best["fuzzy_pct"],
                            "combined_pct": best["combined_pct"]
                        },
                        "total_matches": len(matched_candidates),
                        "matched_elements": [
                            {
                                "text": c["candidate_text"],
                                "combined_pct": c["combined_pct"],
                                "position": f"({c['candidate'].get('x')}, {c['candidate'].get('y')})"
                            } for c in matched_candidates
                        ],
                        "model": match_result["model"],
                        "alpha": alpha,
                        "threshold": threshold
                    }
                )
            else:
                # No match above threshold
                logger.info(f"No semantic match above threshold for text='{request.text}'")
                best = match_result.get("best_candidate")
                
                return AnalyzePageResponse(
                    screenshot_size=screenshot_size,
                    detected_elements=detected_elements,
                    total_text_elements=len(text_elements),
                    total_non_text_elements=len(non_text_elements),
                    matched=False,
                    method="semantic",
                    reason="below_threshold",
                    semantic_result={
                        "best_candidate": {
                            "text": best["candidate_text"] if best else None,
                            "combined_pct": best["combined_pct"] if best else None
                        } if best else None,
                        "threshold": threshold
                    }
                )
            
        # No text search requested - return standard response
        return AnalyzePageResponse(
            screenshot_size=screenshot_size,
            detected_elements=detected_elements,
            total_text_elements=len(text_elements),
            total_non_text_elements=len(non_text_elements)
        )
        
    except Exception as e:
        logger.error(f"Error analyzing page: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _normalize_text_for_lookup(text: str) -> str:
    """Normalize text for direct lookup"""
    if not text:
        return ""
    # Trim and collapse whitespace, case-insensitive
    return " ".join(text.strip().lower().split())


def _perform_direct_lookup(
    query_text: str, 
    element_type: str, 
    elements: List[Dict[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
    """
    Perform direct text lookup with normalization
    
    Returns:
        Tuple of (matched_element, method, reason)
    """
    query_normalized = _normalize_text_for_lookup(query_text)
    
    for element in elements:
        if element.get("type") == element_type:
            element_text = element.get("text", "")
            element_normalized = _normalize_text_for_lookup(element_text)
            
            if element_normalized == query_normalized:
                return element, "direct", None
    
    return None, "direct", "not_found"


def _is_semantic_enabled(config: SemanticConfig, request: AnalyzePageRequest) -> bool:
    """
    Determine if semantic fallback is enabled for this request
    
    Combines server-level config, rollout percentage, and client override
    """
    # Check client override first if allowed
    if config.allow_client_override and request.enable_semantic_match is not None:
        return request.enable_semantic_match
    
    # Check server-level enable flag
    if not config.enable_semantic_match:
        return False
    
    # Check rollout percentage
    if config.semantic_rollout_percent < 100:
        # Deterministic rollout based on request_id or random
        if request.request_id:
            # Use hash of request_id for deterministic decision
            hash_val = int(hashlib.md5(request.request_id.encode()).hexdigest(), 16)
            in_rollout = (hash_val % 100) < config.semantic_rollout_percent
        else:
            # Random rollout
            in_rollout = random.randint(0, 99) < config.semantic_rollout_percent
        
        if not in_rollout:
            return False
    
    return True


@app.post("/find/element", response_model=FindElementResponse)
async def find_element(request: FindElementRequest):
    """
    Find specific element in screenshot based on search criteria
    
    This endpoint analyzes a screenshot and searches for a specific element
    based on text, class name, or element type
    """
    try:
        ai_core: AIVisionCore = app.state.ai_core
        
        # Update confidence threshold if provided
        if request.confidence_threshold:
            ai_core.confidence_threshold = request.confidence_threshold
        
        # Decode base64 image (validates image format)
        img = ai_core.decode_base64_image(request.screenshot_base64)
        
        # Detect UI elements
        detected_elements = ai_core.detect_ui_elements(
            img,
            request.viewport_size.width,
            request.viewport_size.height
        )
        
        # Find element
        found_element = ai_core.find_element(detected_elements, request.search_criteria)
        
        if found_element:
            return FindElementResponse(
                found=True,
                element=found_element,
                message="Element found successfully"
            )
        else:
            return FindElementResponse(
                found=False,
                element=None,
                message="Element not found"
            )
        
    except Exception as e:
        logger.error(f"Error finding element: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/filter/elements")
async def filter_elements(request: FilterElementsRequest):
    """
    Filter detected elements by type or class name
    
    This endpoint analyzes a screenshot and returns filtered elements
    """
    try:
        ai_core: AIVisionCore = app.state.ai_core
        
        # Update confidence threshold if provided
        if request.confidence_threshold:
            ai_core.confidence_threshold = request.confidence_threshold
        
        # Decode base64 image (validates image format)
        img = ai_core.decode_base64_image(request.screenshot_base64)
        
        # Detect UI elements - YOLO accepts numpy array directly
        detected_elements = ai_core.detect_ui_elements(
            img,
            request.viewport_size.width,
            request.viewport_size.height
        )
        
        # Filter elements
        filtered = ai_core.filter_elements(
            detected_elements,
            request.text,
            request.element_type, 
            request.class_name
        )
        
        return {
            "filtered_elements": filtered,
            "count": len(filtered)
        }
        
    except Exception as e:
        logger.error(f"Error filtering elements: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", "8000"))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    logger.info(f"Starting DeepSight AI Vision API on {host}:{port}")
    
    uvicorn.run(
        "deepSightVision_api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

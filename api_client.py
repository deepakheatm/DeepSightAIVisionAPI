"""
DeepSight AI Vision API Client
HTTP client for communicating with DeepSight AI Vision REST API
"""

import base64
import logging
import time
from typing import Dict, Any, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class AIVisionAPIClient:
    """
    HTTP client for DeepSight AI Vision API
    Handles communication, image encoding, error handling, and retries
    """
    
    def __init__(
        self,
        api_base_url: str,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 0.5
    ):
        """
        Initialize DeepSight AI Vision API Client
        
        Args:
            api_base_url: Base URL of DeepSight AI Vision API (e.g., http://localhost:8000)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            backoff_factor: Backoff factor for retries
        """
        self.api_base_url = api_base_url.rstrip('/')
        self.timeout = timeout
        
        # Configure session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info(f"DeepSight AI Vision API Client initialized: {self.api_base_url}")
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode image file to base64 string
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            with open(image_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
        except Exception as e:
            logger.error(f"Error encoding image to base64: {str(e)}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health status
        
        Returns:
            Health status response
        """
        try:
            url = f"{self.api_base_url}/health"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            raise
    
    def analyze_page(
        self,
        screenshot_path: str,
        viewport_width: int,
        viewport_height: int,
        confidence_threshold: Optional[float] = None,
        text: Optional[str] = None,
        element_type: Optional[str] = None,
        enable_semantic_match: Optional[bool] = None,
        semantic_threshold: Optional[float] = None,
        semantic_alpha: Optional[float] = None,
        use_multilingual_model: Optional[bool] = None,
        tenant_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze page screenshot and get complete spatial mapping with optional semantic matching
        
        Args:
            screenshot_path: Path to screenshot image
            viewport_width: Browser viewport width
            viewport_height: Browser viewport height
            confidence_threshold: Optional YOLO confidence threshold
            text: Optional text to search for (enables semantic fallback)
            element_type: Optional element type filter (e.g., "floating_label")
            enable_semantic_match: Enable/disable semantic fallback
            semantic_threshold: Semantic matching threshold (0-100)
            semantic_alpha: Hybrid scoring alpha parameter (0-1)
            use_multilingual_model: Use multilingual model for semantic matching
            tenant_id: Tenant identifier for multi-tenant caching
            request_id: Request identifier for tracking
            
        Returns:
            Complete spatial mapping with all detected elements and semantic results
        """
        try:
            # Encode image to base64
            screenshot_base64 = self.encode_image_to_base64(screenshot_path)
            
            # Prepare request payload
            payload = {
                "screenshot_base64": screenshot_base64,
                "viewport_size": {
                    "width": viewport_width,
                    "height": viewport_height
                }
            }
            
            # Add optional parameters
            if confidence_threshold is not None:
                payload["confidence_threshold"] = confidence_threshold
            
            # Semantic matching parameters
            if text is not None:
                payload["text"] = text
            if element_type is not None:
                payload["element_type"] = element_type
            if enable_semantic_match is not None:
                payload["enable_semantic_match"] = enable_semantic_match
            if semantic_threshold is not None:
                payload["semantic_threshold"] = semantic_threshold
            if semantic_alpha is not None:
                payload["semantic_alpha"] = semantic_alpha
            if use_multilingual_model is not None:
                payload["use_multilingual_model"] = use_multilingual_model
            if tenant_id is not None:
                payload["tenant_id"] = tenant_id
            if request_id is not None:
                payload["request_id"] = request_id
            
            # Make API request
            url = f"{self.api_base_url}/analyze/page"
            
            # Log semantic matching status
            if text and enable_semantic_match:
                logger.info(
                    f"Analyzing page with semantic fallback enabled for text: '{text}' "
                    f"(threshold: {semantic_threshold or 'default'}, "
                    f"multilingual: {use_multilingual_model or False})"
                )
            else:
                logger.info(f"Analyzing page screenshot: {screenshot_path}")
            
            start_time = time.time()
            response = self.session.post(url, json=payload, timeout=self.timeout)
            elapsed_time = time.time() - start_time
            
            response.raise_for_status()
            result = response.json()
            
            # Log results
            log_msg = (
                f"Page analysis completed in {elapsed_time:.2f}s. "
                f"Found {result.get('total_text_elements', 0)} text elements, "
                f"{result.get('total_non_text_elements', 0)} non-text elements"
            )
            
            # Add semantic matching results to log
            if result.get("matched") and result.get("method") == "semantic":
                semantic_result = result.get("semantic_result", {})
                best_candidate = semantic_result.get("best_candidate", {})
                log_msg += (
                    f" | Semantic match: '{best_candidate.get('text', 'N/A')}' "
                    f"(score: {best_candidate.get('combined_pct', 0):.1f}%)"
                )
            
            logger.info(log_msg)
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error analyzing page: {str(e)}")
            raise
    
    def find_element(
        self,
        screenshot_path: str,
        viewport_width: int,
        viewport_height: int,
        text: Optional[str] = None,
        class_name: Optional[str] = None,
        element_type: Optional[str] = None,
        confidence_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Find specific element in screenshot
        
        Args:
            screenshot_path: Path to screenshot image
            viewport_width: Browser viewport width
            viewport_height: Browser viewport height
            text: Text to search for
            class_name: Element class name to search for
            element_type: Element type to search for
            confidence_threshold: Optional YOLO confidence threshold
            
        Returns:
            Search result with found element or None
        """
        try:
            # Encode image to base64
            screenshot_base64 = self.encode_image_to_base64(screenshot_path)
            
            # Prepare search criteria
            search_criteria = {}
            if text:
                search_criteria["text"] = text
            if class_name:
                search_criteria["class_name"] = class_name
            if element_type:
                search_criteria["element_type"] = element_type
            
            # Prepare request payload
            payload = {
                "screenshot_base64": screenshot_base64,
                "viewport_size": {
                    "width": viewport_width,
                    "height": viewport_height
                },
                "search_criteria": search_criteria
            }
            
            if confidence_threshold is not None:
                payload["confidence_threshold"] = confidence_threshold
            
            # Make API request
            url = f"{self.api_base_url}/find/element"
            logger.info(f"Finding element with criteria: {search_criteria}")
            
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            
            if result.get("found"):
                logger.info(f"Element found: {result.get('message')}")
            else:
                logger.warning(f"Element not found: {result.get('message')}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error finding element: {str(e)}")
            raise
    
    def filter_elements(
        self,
        screenshot_path: str,
        viewport_width: int,
        viewport_height: int,
        text: Optional[str] = None,
        element_type: Optional[str] = None,
        class_name: Optional[str] = None,
        confidence_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter detected elements by text, type, or class name
        
        Args:
            screenshot_path: Path to screenshot image
            viewport_width: Browser viewport width
            viewport_height: Browser viewport height
            text: Filter by text content
            element_type: Filter by element type
            class_name: Filter by class name
            confidence_threshold: Optional YOLO confidence threshold
            
        Returns:
            List of filtered elements
        """
        try:
            # Encode image to base64
            screenshot_base64 = self.encode_image_to_base64(screenshot_path)
            
            # Prepare request payload
            payload = {
                "screenshot_base64": screenshot_base64,
                "viewport_size": {
                    "width": viewport_width,
                    "height": viewport_height
                }
            }
            
            # Add filter criteria to payload
            if text:
                payload["text"] = text
            if element_type:
                payload["element_type"] = element_type
            if class_name:
                payload["class_name"] = class_name
            if confidence_threshold is not None:
                payload["confidence_threshold"] = confidence_threshold
            
            # Make API request
            url = f"{self.api_base_url}/filter/elements"
            logger.info(f"Filtering elements with criteria: text={text}, element_type={element_type}, class_name={class_name}")
            
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"Filtered elements count: {result.get('count', 0)}")
            
            return result.get("filtered_elements", [])
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error filtering elements: {str(e)}")
            raise
    
    def close(self):
        """Close the session"""
        self.session.close()
        logger.info("API client session closed")


# ============================================================================
# Convenience Functions
# ============================================================================

def create_api_client(
    api_base_url: str = "http://localhost:8000",
    timeout: int = 30,
    max_retries: int = 3
) -> AIVisionAPIClient:
    """
    Create and return an DeepSight AI Vision API client
    
    Args:
        api_base_url: Base URL of DeepSight AI Vision API
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        
    Returns:
        Configured AIVisionAPIClient instance
    """
    return AIVisionAPIClient(
        api_base_url=api_base_url,
        timeout=timeout,
        max_retries=max_retries
    )


def test_api_connection(api_base_url: str = "http://localhost:8000") -> bool:
    """
    Test connection to DeepSight AI Vision API
    
    Args:
        api_base_url: Base URL of DeepSight AI Vision API
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        client = create_api_client(api_base_url)
        health = client.health_check()
        
        if health.get("status") == "healthy":
            logger.info("API connection test successful")
            return True
        else:
            logger.warning(f"API unhealthy: {health}")
            return False
            
    except Exception as e:
        logger.error(f"API connection test failed: {str(e)}")
        return False
    finally:
        if 'client' in locals():
            client.close()


if __name__ == "__main__":
    # Test the API client
    import sys
    
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    print(f"Testing DeepSight AI Vision API connection to: {api_url}")
    
    if test_api_connection(api_url):
        print("API connection successful")
        sys.exit(0)
    else:
        print("API connection failed")
        sys.exit(1)

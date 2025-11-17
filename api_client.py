"""
DeepSight Vision API Client
HTTP client for communicating with DeepSight Vision REST API
"""

import base64
import logging
import time
from typing import Dict, Any, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class DeepSightAPIClient:
    """
    HTTP client for DeepSight Vision API
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
        Initialize DeepSight Vision API Client
        
        Args:
            api_base_url: Base URL of DeepSight Vision API (e.g., http://localhost:8000)
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
        
        logger.info(f"DeepSight Vision API Client initialized: {self.api_base_url}")
    
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
        confidence_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Analyze page screenshot and get complete spatial mapping
        
        Args:
            screenshot_path: Path to screenshot image
            viewport_width: Browser viewport width
            viewport_height: Browser viewport height
            confidence_threshold: Optional YOLO confidence threshold
            
        Returns:
            Complete spatial mapping with all detected elements
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
            
            if confidence_threshold is not None:
                payload["confidence_threshold"] = confidence_threshold
            
            # Make API request
            url = f"{self.api_base_url}/analyze/page"
            logger.info(f"Analyzing page screenshot: {screenshot_path}")
            
            start_time = time.time()
            response = self.session.post(url, json=payload, timeout=self.timeout)
            elapsed_time = time.time() - start_time
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(
                f"Page analysis completed in {elapsed_time:.2f}s. "
                f"Found {result.get('total_text_elements', 0)} text elements, "
                f"{result.get('total_non_text_elements', 0)} non-text elements"
            )
            
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
) -> DeepSightAPIClient:
    """
    Create and return an DeepSight Vision API client
    
    Args:
        api_base_url: Base URL of DeepSight Vision API
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        
    Returns:
        Configured DeepSightAPIClient instance
    """
    return DeepSightAPIClient(
        api_base_url=api_base_url,
        timeout=timeout,
        max_retries=max_retries
    )


def test_api_connection(api_base_url: str = "http://localhost:8000") -> bool:
    """
    Test connection to DeepSight Vision API
    
    Args:
        api_base_url: Base URL of DeepSight Vision API
        
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
    
    print(f"Testing DeepSight Vision API connection to: {api_url}")
    
    if test_api_connection(api_url):
        print("✓ API connection successful")
        sys.exit(0)
    else:
        print("✗ API connection failed")
        sys.exit(1)


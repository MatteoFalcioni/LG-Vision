from langchain_core.messages import HumanMessage
from state import MultiState
from prompts.mpllry_prompt import prompt
import base64
import requests
import random
import os
import math
from pathlib import Path 

def detect_image_format(image_bytes: bytes) -> tuple[str, str]:
    """
    Detect image format from bytes using magic bytes (file signatures).
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Tuple of (mime_type, extension) e.g., ('image/jpeg', 'jpg')
    """
    # Check magic bytes (most reliable method)
    if image_bytes.startswith(b'\xff\xd8\xff'):
        print("Detected image format: JPEG")
        return 'image/jpeg', 'jpg'
    elif image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
        print("Detected image format: PNG")
        return 'image/png', 'png'
    elif image_bytes.startswith(b'GIF87a') or image_bytes.startswith(b'GIF89a'):
        print("Detected image format: GIF")
        return 'image/gif', 'gif'
    elif image_bytes.startswith(b'RIFF') and b'WEBP' in image_bytes[:12]:
        print("Detected image format: WebP")
        return 'image/webp', 'webp'
    else:
        # Fallback: assume JPEG (most common for Mapillary)
        print("Could not detect image format, defaulting to JPEG")
        return 'image/jpeg', 'jpg'

def encode_b64_from_path(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def encode_b64_paths(file_paths: list[str]) -> list[str]:
    return [encode_b64_from_path(file_path) for file_path in file_paths]

def prepare_multimodal_message(state: MultiState) -> HumanMessage:
    """
    Helper to create multimodal message from state

    Returns:
        message (HumanMessage): The multimodal message
    """
    messages = state.get("messages", [])
    
    # Get text from last message or use default
    if messages and isinstance(messages[-1].content, str):
        text = messages[-1].content
    else:
        text = "Analyze this image"
    
    content_blocks = [{"type": "text", "text": text}]   # it is a list of typed dicts, see https://docs.langchain.com/oss/python/langchain/messages#multimodal
    
    # Add images
    for img_b64 in state.get("images", []):
        content_blocks.append({
            "type": "image",
            "base64": img_b64,
            "mime_type": "image/jpeg"  # NOTE: mpllry images are jpg
        })


    # construct the messages as HumanMessage(content_blocks=...)
    message = HumanMessage(content_blocks=content_blocks)  # v1 format, see https://docs.langchain.com/oss/python/langchain/messages#multimodal
    
    return message   # NOTE: returns msg as is, then you need to wrap it in a list!

def get_multimodal_prompt(good_imgs_paths : list[str], bad_imgs_paths : list[str], text : str = prompt):
    """
    Constructs a multimodal system message, given the textual prompt and images to refer to. 
    The textual prompt defaults to our own custom system prompt.
    """
    content = [{"type": "text", "text": text}]  # start with the prompt
    
    # encode both good and bad images
    good_b64 = encode_b64_paths(good_imgs_paths)
    bad_b64  = encode_b64_paths(bad_imgs_paths)

    # add good images (NOTE: they are png)
    for good_img in good_b64:
        good_text = "This image is acceptable:"
        content.append({"type" : "text", "text" : good_text})
        content.append({"type" : "image", "base64" : good_img, "mime_type" : "image/png"})
    # add bad images
    for bad_img in bad_b64:
        bad_text = "This image is discardable:"
        content.append({"type" : "text", "text" : bad_text})
        content.append({"type" : "image", "base64" : bad_img, "mime_type" : "image/png"})

    system_prompt = HumanMessage(content_blocks=content)
    return system_prompt


def get_mpllry_b64(num_points : int, bbox : list[float] = None, delta = 0.005, max_retries = 10, offset_radius_meters = 50, save_images : bool = False, save_folder : str = None) -> list:
    """
    Leverages the Mapillary API to download images by sampling num_points.
    NOTE: images <= num_points since some points won't have images associated. 
    
    Encodes the images in base 64, and returns a list of the encodings.

    Args:
        num_points: Number of images to retrieve
        bbox: Bounding box as [lat_min, lat_max, lon_min, lon_max]. Defaults to Bologna area.
        delta: Search radius in degrees for initial bbox (roughly 500m for 0.005)
        max_retries: Maximum retry attempts per point with random offset if metadata not found
        offset_radius_meters: Radius in meters for random offset retries (~25m default)
        save_images: If True, save downloaded images to disk
        save_folder: Folder path to save images to (required if save_images=True)

    Returns:
        List of base64-encoded image strings
    """
    # bbox centered on Bologna
    if bbox is None:
        lat_min = 44.4789
        lat_max = 44.5141
        lon_min = 11.3205
        lon_max = 11.3691
    else:
        lat_min, lat_max, lon_min, lon_max = bbox

    # Approximate conversion for offset calculations
    avg_lat = (lat_min + lat_max) / 2
    meters_per_deg_lat = 111000  # meters per degree latitude
    meters_per_deg_lon = 111000 * math.cos(math.radians(avg_lat))  # longitude depends on latitude

    access_token = os.getenv('MAPILLARY_TOKEN')
    if not access_token:
        raise ValueError("MAPILLARY_TOKEN environment variable not set")

    if save_images:
        if save_folder is None:
            raise ValueError("save_folder must be provided when save_images=True")
        save_path = Path(save_folder)
        save_path.mkdir(parents=True, exist_ok=True)

    url = "https://graph.mapillary.com/images"
    base_params = {
        "access_token": access_token,
        "fields": "id,sequence,thumb_1024_url,camera_type,computed_geometry,thumb_original_url",
        "limit": 1
    }

    images_b64 = []
    saved_count = 0
    
    # Sample points uniformly in this bounding box
    for point_idx in range(num_points):
        # Generate base random point
        base_lat = random.uniform(lat_min, lat_max)
        base_lon = random.uniform(lon_min, lon_max)
        
        # Try to find metadata, retrying with offset if not found
        img_metadata = None
        for attempt in range(max_retries):
            # Calculate coordinates (with offset for retries)
            if attempt > 0:
                # Random offset within radius
                angle = random.uniform(0, 2 * math.pi)
                distance = offset_radius_meters * math.sqrt(random.uniform(0, 1))
                
                offset_lat = (distance / meters_per_deg_lat) * math.cos(angle)
                offset_lon = (distance / meters_per_deg_lon) * math.sin(angle)
                
                lat = base_lat + offset_lat
                lon = base_lon + offset_lon
                
                # Ensure we stay within bounds
                lat = max(lat_min, min(lat_max, lat))
                lon = max(lon_min, min(lon_max, lon))
            else:
                lat, lon = base_lat, base_lon
            
            # Construct bbox for Mapillary query
            bbox_str = f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}"
            point_params = {**base_params, "bbox": bbox_str}
            
            try:
                r = requests.get(url, params=point_params, timeout=15)
                
                if r.status_code == 200:
                    data = r.json().get("data", [])
                    if data:
                        img_metadata = data[0]
                        break  # Found metadata, exit retry loop
                    # No data found, continue to next attempt with offset
                # Status not 200, continue to next attempt
            
            except (requests.exceptions.Timeout, requests.exceptions.RequestException):
                # Timeout or other request error, try with offset
                continue
        
        # If we found metadata, try to download the image
        if img_metadata:
            image_url = img_metadata.get('thumb_1024_url')
            if image_url:
                try:
                    img_response = requests.get(image_url, timeout=20)
                    if img_response.status_code == 200:
                        img_content = img_response.content
                        
                        # Detect actual image format from content
                        mime_type, ext = detect_image_format(img_content)
                        
                        # Save image if requested
                        if save_images:
                            # Use image ID from metadata if available, otherwise use index
                            img_id = img_metadata.get('id', f'img_{len(images_b64)}')
                            file_path = save_path / f"{img_id}.{ext}"
                            file_path.write_bytes(img_content)
                            saved_count += 1
                        
                        # Encode to base64
                        img_b64 = base64.b64encode(img_content).decode('utf-8')
                        images_b64.append(img_b64)
                    # If status not 200, just continue to next point (no retry)
                except requests.exceptions.Timeout as e:
                    print(f"Timeout downloading image from {image_url}: {e}")
                    # Continue to next point, no retry
                except requests.exceptions.RequestException as e:
                    print(f"Error downloading image from {image_url}: {e}")
                    # Continue to next point, no retry

    print(f"Downloaded {len(images_b64)} images")
    if save_images:
        print(f"Saved {saved_count} images to {save_folder}")
    return images_b64
        






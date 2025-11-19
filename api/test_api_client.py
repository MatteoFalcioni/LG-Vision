#!/usr/bin/env python3
"""
Script di test per chiamare l'API GroundingSAM da locale
"""
import requests
import time
import sys
from pathlib import Path

SERVER_IP = "local_host"  # use ssh tunneling
SERVER_PORT = 8000
BASE_URL = f"http://{SERVER_IP}:{SERVER_PORT}"

def test_health():
    """Testa se l'API è attiva"""
    print(f"Testing health endpoint at {BASE_URL}/health...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"✓ API is UP: {response.json()}")
        return True
    except Exception as e:
        print(f"✗ API is DOWN: {e}")
        return False

def segment_image(image_path, text_prompt="object", box_threshold=0.35, text_threshold=0.25):
    """Chiama l'API per segmentare un'immagine"""
    
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"✗ File not found: {image_path}")
        return None
    
    file_size_mb = image_path.stat().st_size / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"Sending request:")
    print(f"  Image: {image_path.name} ({file_size_mb:.2f} MB)")
    print(f"  Prompt: '{text_prompt}'")
    print(f"  Thresholds: box={box_threshold}, text={text_threshold}")
    print(f"{'='*60}")
    
    url = f"{BASE_URL}/segment"
    
    start_time = time.time()
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_path.name, f, "image/png")}
            data = {
                "text_prompt": text_prompt,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold
            }
            
            print("\nUploading and processing...")
            response = requests.post(url, files=files, data=data, timeout=300)
            
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ SUCCESS (took {elapsed:.2f}s)")
            print(f"\nResults:")
            print(f"  Detections: {result['num_detections']}")
            print(f"  Labels: {result['labels']}")
            print(f"  Scores: {[f'{s:.2f}' for s in result['scores']]}")
            print(f"  Coverage: {result['mask_coverage']:.4f} ({result['mask_coverage']*100:.2f}%)")
            print(f"  Result saved: {result['result_filename']}")
            print(f"  Server processing time: {result.get('processing_time_seconds', 'N/A')}s")
            print(f"  Total time (including network): {elapsed:.2f}s")
            return result
        else:
            print(f"\n✗ ERROR {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"\n✗ TIMEOUT after {time.time() - start_time:.2f}s")
        return None
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        return None

def main():
    # Test health
    if not test_health():
        print("\nAPI is not responding. Make sure it's running on the server.")
        sys.exit(1)
    
    # Check arguments
    if len(sys.argv) < 2:
        print("\nUsage:")
        print(f"  {sys.argv[0]} <image_path> [text_prompt] [box_threshold] [text_threshold]")
        print("\nExample:")
        print(f"  {sys.argv[0]} image.jpg sky 0.35 0.25")
        sys.exit(1)
    
    image_path = sys.argv[1]
    text_prompt = sys.argv[2] if len(sys.argv) > 2 else "object"
    box_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
    text_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
    
    # Call API
    result = segment_image(image_path, text_prompt, box_threshold, text_threshold)
    
    if result:
        print("\n✓ Test completed successfully!")
    else:
        print("\n✗ Test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()


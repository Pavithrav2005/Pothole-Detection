import os
import sys
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

def quick_test(image_path, model_path=None):
    """
    Quick test for pothole detection on a single image
    
    Args:
        image_path: Path to the image to test
        model_path: Path to the model weights (if None, it will use a default path)
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Load model
    if model_path is None:
        # Try to find the best trained model
        default_model_path = os.path.join(os.getcwd(), "runs/detect/pothole_detector/weights/best.pt")
        if os.path.exists(default_model_path):
            model_path = default_model_path
        else:
            # If no trained model found, use pretrained model
            model_path = "yolov8n.pt"
            print(f"No trained model found, using pretrained model: {model_path}")
    
    # Load the model
    try:
        model = YOLO(model_path)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run inference
    results = model(image_path, conf=0.25)[0]
    
    # Get the original image for display
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get detection results
    boxes = results.boxes.xyxy
    confidences = results.boxes.conf
    
    # Draw bounding boxes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        confidence = confidences[i]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Pothole {confidence:.2f}", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display results
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Pothole Detection Results - Found {len(boxes)} potholes')
    plt.show()
    
    print(f"Found {len(boxes)} potholes in the image.")
    if len(boxes) > 0:
        print(f"Confidence scores: {[float(f'{c:.2f}') for c in confidences]}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_test.py <image_path> [model_path]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    quick_test(image_path, model_path) 
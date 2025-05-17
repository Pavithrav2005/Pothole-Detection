import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image

# Set paths
DATASET_PATH = os.path.join(os.getcwd(), "Dataset")
YAML_PATH = os.path.join(DATASET_PATH, "data.yaml")

def train_model(epochs=50, imgsz=640, batch=16):
    """
    Train a YOLOv8 model on the pothole dataset
    """
    # Initialize a new YOLOv8 model for detection
    model = YOLO("yolov8n.pt")  # Use the nano version for faster training
    
    # Train the model
    results = model.train(
        data=YAML_PATH,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="pothole_detector"
    )
    
    print(f"Training completed. Model saved to {os.getcwd()}/runs/detect/pothole_detector/")
    return model

def validate_model(model=None):
    """
    Validate the trained model on the validation dataset
    """
    if model is None:
        # Load the best trained model
        model_path = os.path.join(os.getcwd(), "runs/detect/pothole_detector/weights/best.pt")
        if os.path.exists(model_path):
            model = YOLO(model_path)
        else:
            print("No trained model found. Please train the model first.")
            return None
    
    # Validate the model
    results = model.val(data=YAML_PATH)
    print(f"Validation mAP50-95: {results.box.map}")
    return results

def detect_potholes_in_image(image_path, model=None, conf_threshold=0.25):
    """
    Detect potholes in a single image
    """
    if model is None:
        # Load the best trained model
        model_path = os.path.join(os.getcwd(), "runs/detect/pothole_detector/weights/best.pt")
        if os.path.exists(model_path):
            model = YOLO(model_path)
        else:
            print("No trained model found. Please train the model first.")
            return None
    
    # Run inference on the image
    results = model(image_path, conf=conf_threshold)[0]
    
    # Get the original image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Draw bounding boxes
    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Pothole {results.boxes.conf[0]:.2f}", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img, results

def detect_potholes_in_video(video_path, output_path=None, model=None, conf_threshold=0.25):
    """
    Detect potholes in a video and save the result
    """
    if model is None:
        # Load the best trained model
        model_path = os.path.join(os.getcwd(), "runs/detect/pothole_detector/weights/best.pt")
        if os.path.exists(model_path):
            model = YOLO(model_path)
        else:
            print("No trained model found. Please train the model first.")
            return
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create output video writer if output path is specified
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection on the frame
        results = model(frame, conf=conf_threshold)[0]
        
        # Draw bounding boxes
        for i, box in enumerate(results.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Pothole {results.boxes.conf[i]:.2f}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Write to output video if specified
        if output_path:
            out.write(frame)
        
        # Display the frame
        cv2.imshow('Pothole Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def main():
    print("Pothole Detection using YOLOv8")
    print("1. Train a new model")
    print("2. Validate the model")
    print("3. Detect potholes in an image")
    print("4. Detect potholes in a video")
    choice = input("Enter your choice (1-4): ")
    
    if choice == '1':
        epochs = int(input("Enter number of epochs (default 50): ") or "50")
        model = train_model(epochs=epochs)
    
    elif choice == '2':
        validate_model()
    
    elif choice == '3':
        image_path = input("Enter the path to the image: ")
        if os.path.exists(image_path):
            model_path = os.path.join(os.getcwd(), "runs/detect/pothole_detector/weights/best.pt")
            if os.path.exists(model_path):
                model = YOLO(model_path)
                img, results = detect_potholes_in_image(image_path, model)
                plt.figure(figsize=(10, 8))
                plt.imshow(img)
                plt.axis('off')
                plt.title('Pothole Detection Results')
                plt.show()
            else:
                print("No trained model found. Please train the model first.")
        else:
            print(f"Image not found at {image_path}")
    
    elif choice == '4':
        video_path = input("Enter the path to the video: ")
        output_path = input("Enter the path for the output video (optional): ")
        if os.path.exists(video_path):
            model_path = os.path.join(os.getcwd(), "runs/detect/pothole_detector/weights/best.pt")
            if os.path.exists(model_path):
                model = YOLO(model_path)
                detect_potholes_in_video(video_path, output_path, model)
            else:
                print("No trained model found. Please train the model first.")
        else:
            print(f"Video not found at {video_path}")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main() 
# Pothole Detection using YOLOv8

This project implements pothole detection on roads using the YOLOv8 model. It leverages a custom dataset with 2105 labeled pothole images.

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Make sure the Dataset folder is correctly structured:
   ```
   Dataset/
   ├── data.yaml
   ├── train/
   │   ├── images/
   │   └── labels/
   └── valid/
       ├── images/
       └── labels/
   ```

## Usage

Run the main script:
```
python pothole_detection.py
```

The script provides four main functionalities:

1. **Train a new model**: Train a YOLOv8 model on the pothole dataset.
2. **Validate the model**: Evaluate the trained model on the validation dataset.
3. **Detect potholes in an image**: Run inference on a single image and display results.
4. **Detect potholes in a video**: Process a video file, detecting potholes in each frame.

## Dataset

The dataset consists of 2105 labeled road images with pothole annotations in YOLO format. The images have been pre-processed and augmented with various transformations for better model generalization.

## Model

The implementation uses YOLOv8n (nano) by default, which offers a good balance between speed and accuracy. The model is trained to detect a single class: potholes.

## Output

- Trained models are saved in the `runs/detect/pothole_detector/weights/` directory.
- For image detection, the results are displayed with bounding boxes around detected potholes.
- For video processing, you can save the output to a new video file with detection results.

## Exit

Press 'q' to exit when viewing video detection results. 
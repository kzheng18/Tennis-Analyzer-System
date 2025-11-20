# Tennis Video Analysis System

“An AI-powered tennis analysis system that automatically detects players, tracks the ball, identifies shot types, measures ball speed, and visualizes shot locations on a virtual mini-court.”

![Alt text](output_video/image_1.png "Optional title")

## Features

**Detection & Tracking**
- YOLO real-time player and ball detection
- ResNet50 court keypoint detection (14 points)
- Position-based player ID normalization 
- Trajectory smoothing with gap interpolation

**Analytics**
- Shot detection via velocity analysis
- Ball speed calculation with perspective correction
- Shot type classification: Topspin, Flat, Slice 
- Bounce detection with in/out determination
- Mini-court tactical visualization

## Technical Stack

**Models**
- **ResNet50**: Custom trained on public dataset from google drive for court geometry
- **YOLOV12**: Ultralytics pre-trained for player detection
- **YOLOV5**: Custom-trained on Roboflow for ball detection


**Libraries**: PyTorch, OpenCV, Pandas, NumPy, Ultralytics

**Model Training**: ResNet50 trained on Roboflow | YOLO from Ultralytics COCO

## Future Tuning
### 1. Mini-Court Coordinate Accuracy
**Issue**: Ball/player positions may appear offset on mini-court visualization  
**Solution**: Adjust coordinate transformation thresholds

### 2. Ball In/Out Call Accuracy
**Issue**: False positives/negatives on bounce detection  
**Solution**: Fine-tune singles court boundaries and margin thresholds

### 3. Player Tracking Stability
**Issue**: Players disappear or IDs swap during rallies  
**Solution**: Adjust YOLO confidence thresholds and normalization parameters

### 4. Ball Speed Calculation
**Issue**: Speed readings inconsistent or inaccurate  
**Solution**: Calibrate perspective correction factor

### 5. Shot Type Classification
**Issue**: Too many/few Topspin or Slice detections  
**Solution**: Adjust trajectory analysis thresholds


## Slide Show
[AI-driven Tennis Analytics From a Single Camera Perspective](https://docs.google.com/presentation/d/1fhx5SRMFeWyEuho4I2ZenzR7ZMevnlG8E6GQp0nxsMY/edit?usp=sharing)
# DOOH AR Senior D

A computer vision project for detecting landmark buildings in images using YOLO object detection and segmentation models. The project includes both AI model training/inference capabilities and a web-based demo application.

## Project Structure

```
DOOH-AR-Senior-D/
├── dooh_ai/           # AI/ML components (model training, inference, notebooks)
│   ├── models/        # Trained models and weights
│   ├── test_images/   # Test images for inference
│   ├── scripts/       # Utility scripts (e.g., HEIC to JPG conversion)
│   └── yolov8_test.ipynb  # Jupyter notebook for model development
└── dooh_web/          # Web-based demo application
    ├── index.html     # Web interface
    └── script.js      # ONNX Runtime Web integration
```

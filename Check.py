import torch
from ultralytics import YOLO

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    print("CUDA is available")
    # Assuming you've loaded your YOLO model, check the device it's running on
    model = YOLO("YoLo_Weight/yolov8n.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model is running on: {device}")
else:
    print("CUDA is not available, running on CPU")
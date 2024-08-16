from ultralytics import YOLO

# Load the model.
model = YOLO('datasets/Yolov8_cardboard_dataset/yolov8n.pt')

# Training.
results = model.train(
    data='data.yaml',
    imgsz=640,
    epochs=50,
    batch=4,
    name='yolov8n_v8_50e'
)
from ultralytics import YOLO

def train_model():
    model = YOLO("yolo11n.pt")
    results = model.train(data='datasets/data.yaml', epochs=100, batch=16)
train_model()

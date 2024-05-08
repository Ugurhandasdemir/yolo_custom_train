from ultralytics import YOLO
dataset ="C:/Users/ugurh/PycharmProjects/Yolo_train/dataset.yaml"

# Load a model
model = YOLO("C:/Users/ugurh/PycharmProjects/Yolo_train/yolov8m.pt")  

# Train the model
results = model.train(data=dataset, epochs=100, imgsz=800, pretrained=True, save_period=1, plots=True, device='0')

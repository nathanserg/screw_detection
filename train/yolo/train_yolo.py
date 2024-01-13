from ultralytics import YOLO

save_dir = 'train/yolo'

model = YOLO("train/yolov8n.pt")

results = model.train(data='train/config.yaml', imgsz=640, epochs=50, batch=8, name='yolov8n_custom', save_dir=save_dir)
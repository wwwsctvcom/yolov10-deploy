from ultralytics import YOLO


if __name__ == "__main__":
    # Load YOLOv10n model from scratch
    model = YOLO("yolov10n.yaml")

    # Train the model
    model.train(data="coco128.yaml", epochs=10, imgsz=640)

from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("yolov10n.pt")
    model.export(format="onnx")

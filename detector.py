from ultralytics import YOLO


class ObjectDetector:

    def __init__(self, model_name="yolo11n.pt"):
        self.model = YOLO(model_name)

    def detect(self, frame, confidence=0.5):
        results = self.model(
            frame,
            conf=confidence,
            verbose=False
        )
        return results[0]
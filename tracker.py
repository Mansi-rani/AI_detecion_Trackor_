from ultralytics import YOLO

class ObjectTracker:

    def __init__(self, model_name="yolo11n.pt"):
        self.model = YOLO(model_name)

    def track(self, frame, confidence=0.5):
        results = self.model.track(
            frame,
            conf=confidence,
            persist=True,
            verbose=False
        )
        return results[0]
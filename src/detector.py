from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class Detector:
    """
    YOLOv8m wrapper for person detection.
    Returns detections as np.ndarray of shape (N, 6): [x1, y1, x2, y2, conf, cls]
    """

    def __init__(self, cfg: dict):
        det_cfg = cfg["detector"]
        weights_dir = Path(cfg["paths"]["weights_dir"])
        weights_dir.mkdir(parents=True, exist_ok=True)

        model_path = weights_dir / det_cfg["model"]
        self.model = YOLO(str(model_path))  # auto-downloads if not present

        self.device = det_cfg["device"]
        self.conf = det_cfg["conf_threshold"]
        self.iou = det_cfg["iou_threshold"]
        self.target_classes = det_cfg["target_classes"]

        # Warm up
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._infer(dummy)

    def _infer(self, frame: np.ndarray):
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            classes=self.target_classes,
            device=self.device,
            verbose=False,
        )
        return results

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Args:
            frame: BGR image as np.ndarray (H, W, 3)
        Returns:
            detections: np.ndarray (N, 6) — [x1, y1, x2, y2, conf, cls]
                        Empty array of shape (0, 6) if no detections.
        """
        results = self._infer(frame)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        xyxy = boxes.xyxy.cpu().numpy()       # (N, 4)
        conf = boxes.conf.cpu().numpy()       # (N,)
        cls  = boxes.cls.cpu().numpy()        # (N,)

        detections = np.concatenate(
            [xyxy, conf[:, None], cls[:, None]], axis=1
        ).astype(np.float32)

        return detections
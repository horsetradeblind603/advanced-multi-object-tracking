from pathlib import Path

import cv2
import numpy as np
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class VideoWriter:
    """
    Assembles annotated frames into an MP4.
    FPS is taken from the sequence loader; codec from config.
    """

    def __init__(self, cfg: dict, output_path: Path,
                 fps: float, frame_size: tuple):
        """
        Args:
            output_path : full path to output .mp4
            fps         : frames per second (from seqinfo.ini)
            frame_size  : (width, height)
        """
        codec     = cfg["video"]["codec"]
        fourcc    = cv2.VideoWriter_fourcc(*codec)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, frame_size)
        self._path = output_path

    def write(self, frame: np.ndarray):
        self._writer.write(frame)

    def release(self):
        self._writer.release()
        print(f"  Video saved: {self._path}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
from collections import defaultdict

import cv2
import numpy as np

from src.tracker import TrackState

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
THICKNESS  = 2

# State border styles: (color_BGR, line_type)
STATE_STYLE = {
    TrackState.BORN  : ((0,   255, 255), cv2.LINE_4),   # yellow  solid
    TrackState.ACTIVE: ((0,   255, 0  ), cv2.LINE_8),   # green   solid
    TrackState.LOST  : ((0,   140, 255), cv2.LINE_4),   # orange  thin
    TrackState.DEAD  : ((0,   0,   200), cv2.LINE_4),   # red     thin
}

STATE_LABEL = {
    TrackState.BORN  : "B",
    TrackState.ACTIVE: "A",
    TrackState.LOST  : "L",
    TrackState.DEAD  : "D",
}


def _id_color(track_id: int) -> tuple:
    """Deterministic, visually distinct color per track ID."""
    np.random.seed(track_id * 7 + 13)
    return tuple(int(c) for c in np.random.randint(80, 230, 3))


class Visualizer:
    """
    Draws bounding boxes, track IDs, lifecycle state labels,
    and velocity vectors onto frames.

    Maintains a short position history per track ID to compute velocity.
    """

    def __init__(self, cfg: dict):
        vis_cfg = cfg["visualization"]
        self.min_vel    = vis_cfg["velocity_min_magnitude"]
        self.arrow_scale = vis_cfg["velocity_arrow_scale"]
        self.show_state  = vis_cfg["show_state_label"]
        self.thickness   = cfg.get("visualization", {}).get(
            "box_thickness", THICKNESS)

        # track_id -> deque of (cx, cy) over last 2 frames
        self._prev_centers: dict[int, tuple] = {}

    def draw(self, frame: np.ndarray, tracks: list) -> np.ndarray:
        """
        Args:
            frame  : BGR image
            tracks : list of track dicts from TrackerWrapper.update()
        Returns:
            Annotated frame (copy)
        """
        vis = frame.copy()

        for t in tracks:
            if t["bbox"] is None:
                # Lost/dead — no box to draw, clean up velocity history
                self._prev_centers.pop(t["track_id"], None)
                continue

            tid   = t["track_id"]
            state = t["state"]
            x1, y1, x2, y2 = t["bbox"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            color, line_type = STATE_STYLE.get(
                state, STATE_STYLE[TrackState.ACTIVE])
            id_color = _id_color(tid)

            # Bounding box — ID color fill, state color border
            cv2.rectangle(vis, (x1, y1), (x2, y2),
                          id_color, self.thickness, line_type)

            # Label: ID + state initial
            state_ch = STATE_LABEL.get(state, "?")
            label    = f"ID{tid}"
            if self.show_state:
                label += f" [{state_ch}]"

            (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1),
                          id_color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                        FONT, FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)

            # Velocity vector
            if tid in self._prev_centers:
                px, py = self._prev_centers[tid]
                dx = cx - px
                dy = cy - py
                mag = np.sqrt(dx * dx + dy * dy)
                if mag >= self.min_vel:
                    ex = int(cx + dx * self.arrow_scale)
                    ey = int(cy + dy * self.arrow_scale)
                    cv2.arrowedLine(vis, (cx, cy), (ex, ey),
                                    id_color, 2,
                                    cv2.LINE_AA, tipLength=0.35)

            self._prev_centers[tid] = (cx, cy)

        return vis

    def reset(self):
        """Call between sequences."""
        self._prev_centers.clear()
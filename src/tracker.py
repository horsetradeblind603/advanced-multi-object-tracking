from pathlib import Path

import numpy as np
import yaml
from boxmot import StrongSort


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class TrackState:
    BORN   = "born"
    ACTIVE = "active"
    LOST   = "lost"
    DEAD   = "dead"


class TrackerWrapper:
    """
    StrongSORT + OSNet wrapper.
    Manages track lifecycle states on top of boxmot's raw tracker output.

    Output per frame: list of dicts, one per active track:
        {
            "track_id"  : int,
            "bbox"      : [x1, y1, x2, y2],
            "conf"      : float,
            "state"     : str,   # born | active | lost | dead
            "frame_id"  : int,
            "age"       : int,   # frames since first seen
        }
    """

    def __init__(self, cfg: dict):
        ss_cfg       = cfg["strongsort"]
        weights_dir  = Path(cfg["paths"]["weights_dir"])
        weights_dir.mkdir(parents=True, exist_ok=True)

        osnet_path = weights_dir / ss_cfg["osnet_model"]

        self.tracker = StrongSort(
            reid_weights=osnet_path,
            device=ss_cfg["device"],
            half=False,
            max_age=ss_cfg["max_age"],
            min_hits=ss_cfg["min_hits"],
            iou_threshold=ss_cfg["iou_threshold"],
            ema_alpha=ss_cfg["ema_alpha"],
            mc_lambda=ss_cfg["mc_lambda"],
        )


        self.min_hits = ss_cfg["min_hits"]

        # Lifecycle tracking
        # track_id -> {"first_frame": int, "hit_count": int, "last_seen": int}
        self._track_registry: dict[int, dict] = {}
        self._dead_ids:        set[int]        = set()

    def update(self, dets: np.ndarray, frame: np.ndarray,
               frame_id: int) -> list[dict]:
        """
        Args:
            dets     : (N, 6) [x1, y1, x2, y2, conf, cls] — from Detector
            frame    : BGR image (H, W, 3)
            frame_id : 1-indexed frame number

        Returns:
            List of track dicts for this frame.
        """
        if dets.shape[0] == 0:
            # Still need to call update to age existing tracks
            empty = np.empty((0, 6), dtype=np.float32)
            raw   = self.tracker.update(empty, frame)
        else:
            raw = self.tracker.update(dets, frame)

        tracks = []

        if raw is None or len(raw) == 0:
            # Mark previously seen tracks as lost or dead
            self._age_missing(frame_id, tracks)
            return tracks

        active_ids = set()

        for row in raw:
            # boxmot StrongSORT output: [x1, y1, x2, y2, track_id, conf, cls, ...]
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            tid             = int(row[4])
            conf            = float(row[5])
            active_ids.add(tid)

            if tid not in self._track_registry:
                self._track_registry[tid] = {
                    "first_frame": frame_id,
                    "hit_count"  : 1,
                    "last_seen"  : frame_id,
                }
                state = TrackState.BORN
            else:
                reg = self._track_registry[tid]
                reg["hit_count"] += 1
                reg["last_seen"]  = frame_id
                state = (TrackState.ACTIVE
                         if reg["hit_count"] > self.min_hits
                         else TrackState.BORN)

            age = frame_id - self._track_registry[tid]["first_frame"] + 1

            tracks.append({
                "track_id" : tid,
                "bbox"     : [x1, y1, x2, y2],
                "conf"     : conf,
                "state"    : state,
                "frame_id" : frame_id,
                "age"      : age,
            })

        # Handle tracks not returned this frame → lost or dead
        self._age_missing(frame_id, tracks, active_ids)

        return tracks

    def _age_missing(self, frame_id: int, tracks: list,
                     active_ids: set | None = None):
        """
        Tracks in registry but absent from current output are marked lost.
        Tracks unseen for longer than max_age are marked dead.
        """
        if active_ids is None:
            active_ids = set()

        max_age = self.tracker.max_age

        for tid, reg in self._track_registry.items():
            if tid in active_ids or tid in self._dead_ids:
                continue

            frames_missing = frame_id - reg["last_seen"]

            if frames_missing > max_age:
                self._dead_ids.add(tid)
                tracks.append({
                    "track_id" : tid,
                    "bbox"     : None,
                    "conf"     : 0.0,
                    "state"    : TrackState.DEAD,
                    "frame_id" : frame_id,
                    "age"      : frame_id - reg["first_frame"] + 1,
                })
            else:
                tracks.append({
                    "track_id" : tid,
                    "bbox"     : None,
                    "conf"     : 0.0,
                    "state"    : TrackState.LOST,
                    "frame_id" : frame_id,
                    "age"      : frame_id - reg["first_frame"] + 1,
                })

    def reset(self):
        """Call between sequences."""
        self.tracker.reset()
        self._track_registry.clear()
        self._dead_ids.clear()
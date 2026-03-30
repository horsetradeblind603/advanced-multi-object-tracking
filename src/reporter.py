import json
import time
from collections import defaultdict
from pathlib import Path

import yaml

from src.tracker import TrackState


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class Reporter:
    """
    Accumulates per-frame tracking data over a full sequence run
    and writes a structured JSON report to outputs/reports/.

    Call .update() each frame, .save() at sequence end.
    """

    def __init__(self, cfg: dict, seq_name: str):
        self.seq_name   = seq_name
        self.output_dir = Path(cfg["paths"]["reports_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Per-track accumulators
        # track_id -> {first_frame, last_frame, states_seen,
        #               confidences, centroids}
        self._tracks: dict[int, dict] = {}

        # Per-frame counts for density stats
        self._frame_active_counts: list[int] = []

        # ID switch detection:
        # crude proxy — new born IDs appearing after frame warmup
        self._id_switch_events: list[dict] = []
        self._prev_active_ids: set         = set()

        # Timing
        self._frame_times: list[float] = []
        self._total_detections          = 0
        self._frame_count               = 0
        self._start_wall                = time.time()

    def update(self, tracks: list, frame_id: int,
               det_count: int, frame_time_ms: float):
        """
        Call once per frame after tracker.update().

        Args:
            tracks        : list of track dicts from TrackerWrapper
            frame_id      : 1-indexed frame number
            det_count     : number of raw detections this frame
            frame_time_ms : total detect+track time in ms
        """
        self._frame_count      += 1
        self._total_detections += det_count
        self._frame_times.append(frame_time_ms)

        active_ids = set()

        for t in tracks:
            tid   = t["track_id"]
            state = t["state"]

            if tid not in self._tracks:
                self._tracks[tid] = {
                    "first_frame" : frame_id,
                    "last_frame"  : frame_id,
                    "states_seen" : set(),
                    "confidences" : [],
                    "centroids"   : [],
                }

            rec = self._tracks[tid]
            rec["last_frame"] = frame_id
            rec["states_seen"].add(state)

            if t["conf"] > 0:
                rec["confidences"].append(t["conf"])

            if t["bbox"] is not None:
                x1, y1, x2, y2 = t["bbox"]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                rec["centroids"].append([cx, cy])

            if state in (TrackState.ACTIVE, TrackState.BORN):
                active_ids.add(tid)

        # ID switch proxy: born IDs appearing after frame 10
        # that weren't previously seen
        if frame_id > 10:
            all_known = set(self._tracks.keys())
            new_ids   = active_ids - self._prev_active_ids - all_known
            # Re-check: genuinely new IDs this frame past warmup
            new_born = {
                tid for tid in active_ids
                if tid not in self._prev_active_ids
                and self._tracks.get(tid, {}).get("first_frame", 0) == frame_id
                and frame_id > 10
            }
            if new_born:
                self._id_switch_events.append({
                    "frame_id" : frame_id,
                    "new_ids"  : list(new_born),
                })

        self._frame_active_counts.append(len(active_ids))
        self._prev_active_ids = active_ids

    def save(self) -> Path:
        """
        Finalize and write the JSON report.
        Returns the path of the written file.
        """
        per_track = []
        for tid, rec in self._tracks.items():
            lifetime = rec["last_frame"] - rec["first_frame"] + 1
            mean_conf = (sum(rec["confidences"]) / len(rec["confidences"])
                         if rec["confidences"] else 0.0)

            # Trajectory centroid = mean of all per-frame centroids
            if rec["centroids"]:
                xs = [c[0] for c in rec["centroids"]]
                ys = [c[1] for c in rec["centroids"]]
                traj_centroid = [round(sum(xs)/len(xs), 1),
                                 round(sum(ys)/len(ys), 1)]
            else:
                traj_centroid = [None, None]

            per_track.append({
                "track_id"       : tid,
                "first_frame"    : rec["first_frame"],
                "last_frame"     : rec["last_frame"],
                "lifetime_frames": lifetime,
                "states_seen"    : sorted(rec["states_seen"]),
                "mean_conf"      : round(mean_conf, 4),
                "traj_centroid"  : traj_centroid,
            })

        # Sort by first appearance
        per_track.sort(key=lambda x: x["first_frame"])

        lifetimes = [t["lifetime_frames"] for t in per_track]
        avg_lifetime = (sum(lifetimes) / len(lifetimes)
                        if lifetimes else 0.0)

        avg_active = (sum(self._frame_active_counts)
                      / len(self._frame_active_counts)
                      if self._frame_active_counts else 0.0)

        avg_ms  = (sum(self._frame_times) / len(self._frame_times)
                   if self._frame_times else 0.0)
        avg_fps = 1000 / avg_ms if avg_ms > 0 else 0.0

        report = {
            "sequence"       : self.seq_name,
            "summary"        : {
                "total_frames"          : self._frame_count,
                "total_detections"      : self._total_detections,
                "unique_track_ids"      : len(self._tracks),
                "avg_active_per_frame"  : round(avg_active, 2),
                "avg_track_lifetime_frames": round(avg_lifetime, 1),
                "id_switch_events"      : len(self._id_switch_events),
                "avg_inference_ms"      : round(avg_ms, 2),
                "avg_fps"               : round(avg_fps, 2),
                "wall_time_seconds"     : round(
                    time.time() - self._start_wall, 1),
            },
            "id_switch_log"  : self._id_switch_events,
            "tracks"         : per_track,
        }

        out_path = self.output_dir / f"{self.seq_name}_report.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

        return out_path
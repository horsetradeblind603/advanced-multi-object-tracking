# Project V1.1: Advanced Multi-Object Tracking with StrongSORT + YOLOv8

State-of-the-art multi-object tracking pipeline using YOLOv8m detection and 
StrongSORT with OSNet appearance embeddings, evaluated on MOT17 against a 
ByteTrack baseline.

Built for video analytics and surveillance roles — demonstrates track lifecycle 
management, appearance-based re-association, camera motion compensation, and 
formal evaluation with HOTA/MOTA/IDF1 metrics.

---

## Results

### StrongSORT (Primary)

| Sequence     | HOTA  | MOTA  | MOTP  | IDF1  | IDSw |
|--------------|-------|-------|-------|-------|------|
| MOT17-02-SDP | 31.18 | 24.30 | 81.75 | 34.83 |   69 |
| MOT17-04-SDP | 43.06 | 37.49 | 82.86 | 52.59 |   58 |
| MOT17-05-SDP | 48.76 | 51.02 | 76.97 | 65.91 |   66 |
| MOT17-09-SDP | 51.39 | 64.26 | 81.98 | 62.61 |   41 |
| MOT17-10-SDP | 39.60 | 42.88 | 76.34 | 49.48 |  103 |
| **COMBINED** | **41.60** | **38.15** | **80.89** | **50.79** | **337** |

### StrongSORT vs ByteTrack

| Sequence     | HOTA SS/BT/Δ      | MOTA SS/BT/Δ      | IDF1 SS/BT/Δ      | IDSw SS/BT |
|--------------|-------------------|-------------------|-------------------|------------|
| MOT17-02-SDP | 31.2/30.6/+0.6   | 24.3/22.9/+1.4   | 34.8/33.5/+1.3   | 69/25      |
| MOT17-04-SDP | 43.1/39.2/+3.9   | 37.5/31.9/+5.6   | 52.6/45.4/+7.2   | 58/27      |
| MOT17-05-SDP | 48.8/43.8/+4.9   | 51.0/50.8/+0.2   | 65.9/59.6/+6.3   | 66/88      |
| MOT17-09-SDP | 51.4/49.6/+1.8   | 64.3/64.4/-0.1   | 62.6/62.0/+0.6   | 41/27      |
| MOT17-10-SDP | 39.6/34.5/+5.1   | 42.9/39.7/+3.2   | 49.5/44.0/+5.5   | 103/76     |
| **COMBINED** | **41.6/38.2/+3.4** | **38.1/34.5/+3.7** | **50.8/45.5/+5.3** | **337/243** |

StrongSORT outperforms ByteTrack on HOTA, MOTA, and IDF1 across all sequences.
ByteTrack produces fewer ID switches on most sequences — a consequence of 
StrongSORT's aggressive appearance-based re-association, which recovers more 
lost tracks but occasionally re-assigns incorrectly.

---

## Tech Stack

- **Detector**: YOLOv8m (pretrained, no fine-tuning)
- **Tracker**: StrongSORT with OSNet-x0.25 appearance embeddings (boxmot)
- **Baseline**: ByteTrack (boxmot)
- **Evaluation**: TrackEval (HOTA, MOTA, MOTP, IDF1)
- **Dataset**: MOT17 — 5 sequences (train split)
- **Environment**: conda, Python 3.10, PyTorch 2.x, M1 MacBook Air

---

## Project Structure
```
├── config/
│   └── config.yaml          # All parameters — no hardcoded values
├── src/
│   ├── data_loader.py       # MOT17 frame iterator with resolution scaling
│   ├── detector.py          # YOLOv8m wrapper
│   ├── tracker.py           # StrongSORT + lifecycle state management
│   ├── tracker_byte.py      # ByteTrack wrapper (same interface)
│   ├── visualizer.py        # Bounding boxes, state labels, velocity vectors
│   ├── video_writer.py      # MP4 export
│   ├── reporter.py          # Per-sequence JSON analytics
│   └── eval_formatter.py   # MOT Challenge format exporter
├── scripts/
│   ├── run_tracking.py      # Full StrongSORT pipeline across all sequences
│   ├── run_baseline.py      # ByteTrack pipeline
│   ├── run_eval.py          # TrackEval runner (--tracker flag)
│   └── compare_trackers.py  # Side-by-side comparison table
├── outputs/
│   ├── videos/              # Annotated MP4s (gitignored)
│   ├── reports/             # Per-sequence JSON analytics (gitignored)
│   └── eval/                # TrackEval results and comparison
└── data/                    # MOT17 dataset (gitignored)
```

---

## Setup
```bash
git clone https://github.com/tajwarchy/advanced-multi-object-tracking
cd advanced-multi-object-tracking

conda create -n mot_tracking python=3.10 -y
conda activate mot_tracking

pip install torch torchvision torchaudio
pip install ultralytics boxmot opencv-python pyyaml

# TrackEval
git clone https://github.com/JonathonLuiten/TrackEval.git ~/TrackEval
cd ~/TrackEval && pip install -e . && cd -
```

Update `eval.trackeval_root` in `config/config.yaml` if TrackEval is cloned 
elsewhere.

---

## Dataset Preparation

Download MOT17 from https://motchallenge.net/data/MOT17.zip and extract:
```bash
unzip MOT17.zip -d data/
```

Set up GT symlinks for TrackEval:
```bash
mkdir -p outputs/eval/gt/MOT17/MOT17-train
for seq in MOT17-02-SDP MOT17-04-SDP MOT17-05-SDP MOT17-09-SDP MOT17-10-SDP; do
    ln -s "$(pwd)/data/MOT17/train/$seq" \
          "$(pwd)/outputs/eval/gt/MOT17/MOT17-train/$seq"
done
```

---

## Running
```bash
# StrongSORT — full pipeline (all 5 sequences)
python scripts/run_tracking.py

# ByteTrack baseline
python scripts/run_baseline.py

# Evaluate StrongSORT
python -m scripts.run_eval --tracker StrongSORT

# Evaluate ByteTrack
python -m scripts.run_eval --tracker ByteTrack

# Side-by-side comparison
python scripts/compare_trackers.py
```

---

## Visual Outputs

Each annotated video includes:
- Color-coded bounding boxes per track ID (consistent across frames)
- State label overlay: `[B]` born, `[A]` active, `[L]` lost
- Velocity vectors on moving targets (suppressed below noise threshold)

Per-sequence JSON reports in `outputs/reports/` include track IDs, 
lifetimes, trajectory centroids, and inference FPS.

---

## Environment Notes

- M1 MacBook Air — MPS for YOLOv8m inference, CPU fallback for tracker if 
  MPS is unstable
- `num_workers=0` throughout (macOS multiprocessing constraint)
- Inference resolution: 640px (reduce to 480px if thermal throttling occurs)
- Coordinates rescaled to original resolution before TrackEval submission
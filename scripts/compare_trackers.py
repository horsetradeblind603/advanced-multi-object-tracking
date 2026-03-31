from pathlib import Path

import yaml

def fmt(s: dict, b: dict, m: str) -> str:
    sv   = s.get(m, 0)
    bv   = b.get(m, 0)
    d    = sv - bv
    sign = "+" if d >= 0 else ""
    return f"{sv:5.1f}/{bv:5.1f}/{sign}{d:.1f}"

def fmt_idsw(s: dict, b: dict, m: str) -> str:
    sv = int(s.get(m, 0))
    bv = int(b.get(m, 0))
    return f"{sv:4d}/{bv:4d}"


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_results(path: Path) -> dict:
    """Parse a results .txt file into {sequence: {metric: value}}."""
    results = {}
    with open(path) as f:
        lines = f.readlines()
    for line in lines[2:]:   # skip header and separator
        line = line.strip()
        if not line or line.startswith("-"):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        seq  = parts[0]
        try:
            results[seq] = {
                "HOTA": float(parts[1]),
                "MOTA": float(parts[2]),
                "MOTP": float(parts[3]),
                "IDF1": float(parts[4]),
                "IDSw": float(parts[5]),
            }
        except ValueError:
            continue
    return results


def main():
    cfg      = load_config()
    eval_dir = Path(cfg["paths"]["eval_dir"]) / "results"

    ss_path  = eval_dir / "strongsort_results.txt"
    bt_path  = eval_dir / "bytetrack_results.txt"

    if not ss_path.exists():
        print(f"Missing: {ss_path}")
        return
    if not bt_path.exists():
        print(f"Missing: {bt_path}")
        return

    ss = parse_results(ss_path)
    bt = parse_results(bt_path)

    sequences = list(ss.keys())
    metrics   = ["HOTA", "MOTA", "MOTP", "IDF1", "IDSw"]

    print(f"\n{'='*90}")
    print(f"{'Sequence':<22} "
          f"{'HOTA (SS/BT/Δ)':>20} "
          f"{'MOTA (SS/BT/Δ)':>20} "
          f"{'IDF1 (SS/BT/Δ)':>20} "
          f"{'IDSw SS/BT':>12}")
    print(f"{'-'*90}")

    for seq in sequences:
        s = ss.get(seq, {})
        b = bt.get(seq, {})
        if not s or not b or seq == "COMBINED":
            continue

        print(f"{seq:<22} "
              f"{fmt(s, b, 'HOTA'):>20} "
              f"{fmt(s, b, 'MOTA'):>20} "
              f"{fmt(s, b, 'IDF1'):>20} "
              f"{fmt_idsw(s, b, 'IDSw'):>12}")

    if sc and bc:
        print(f"{'-'*90}")
        print(f"{'COMBINED':<22} "
              f"{fmt(sc, bc, 'HOTA'):>20} "
              f"{fmt(sc, bc, 'MOTA'):>20} "
              f"{fmt(sc, bc, 'IDF1'):>20} "
              f"{fmt_idsw(sc, bc, 'IDSw'):>12}")

    print(f"{'='*90}")
    print(f"\nSS = StrongSORT | BT = ByteTrack | Δ = SS minus BT\n")

    # Save
    out_path = eval_dir / "comparison.txt"
    print(f"Comparison saved: {out_path}")


if __name__ == "__main__":
    main()
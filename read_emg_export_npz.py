# read_emg_export_npz.py
# Usage:
#   python read_emg_export_npz.py path/to/export_processed.npz
#   python read_emg_export_npz.py path/to/export_processed.npz --plot
#
# Expects keys:
#   emg: (n_t, 6, 8, 8)
#   aux:      (n_t, k) or empty (0,0)
#   bad_mask_grid: (6, 8, 8) bool
#   time_s:   (n_t,)
#   meta:     JSON string (saved via json.dumps)

import argparse
import json
import numpy as np

def _decode_meta(meta_obj):
    """meta was saved as json.dumps(meta). It may load as 0-d ndarray of dtype <U or object."""
    if meta_obj is None:
        return {}
    try:
        s = meta_obj.item() if hasattr(meta_obj, "item") else meta_obj
        if isinstance(s, bytes):
            s = s.decode("utf-8", errors="replace")
        if isinstance(s, str):
            return json.loads(s)
        return {"_raw_meta": str(s)}
    except Exception as e:
        return {"_meta_decode_error": str(e), "_raw_meta": str(meta_obj)}

def load_export_npz(path: str):
    z = np.load(path, allow_pickle=False)

    # Backwards/forwards compatible fetching
    if "emg" in z.files:
        emg_grid = z["emg"]
    else:
        raise KeyError("Missing key 'emg' in NPZ.")

    aux = z["aux"] if "aux" in z.files else np.empty((0, 0))
    bad_mask_grid = z["bad_mask_grid"] if "bad_mask_grid" in z.files else None
    time_s = z["time_s"] if "time_s" in z.files else None
    meta = _decode_meta(z["meta"]) if "meta" in z.files else {}

    return emg_grid, aux, bad_mask_grid, time_s, meta

def summarize(emg_grid, aux, bad_mask_grid, time_s, meta):
    print("=== META ===")
    if meta:
        for k, v in meta.items():
            print(f"{k}: {v}")
    else:
        print("(no meta)")

    print("\n=== ARRAYS ===")
    print("emg      :", emg_grid.shape, emg_grid.dtype)
    print("aux           :", aux.shape, aux.dtype)
    if bad_mask_grid is None:
        print("bad_mask_grid :", None)
    else:
        print("bad_mask_grid :", bad_mask_grid.shape, bad_mask_grid.dtype)
    if time_s is None:
        print("time_s        :", None)
    else:
        print("time_s        :", time_s.shape, time_s.dtype)

    print("\n=== SANITY CHECKS ===")
    n_t = emg_grid.shape[0]
    if time_s is not None and time_s.shape[0] != n_t:
        print(f"[WARN] time_s length ({time_s.shape[0]}) != emg length ({n_t})")
    if aux.size != 0 and aux.shape[0] != n_t:
        print(f"[WARN] aux length ({aux.shape[0]}) != emg length ({n_t})")
    if bad_mask_grid is not None:
        if bad_mask_grid.shape != (6, 8, 8):
            print(f"[WARN] bad_mask_grid shape is {bad_mask_grid.shape}, expected (6, 8, 8)")
        bad_counts = bad_mask_grid.reshape(6, -1).sum(axis=1)
        print("bad electrodes per grid:", bad_counts.tolist(), "(out of 64 each)")

    print("\n=== QUICK STATS ===")
    print("emg abs mean:", float(np.mean(np.abs(emg_grid))))
    print("emg abs max :", float(np.max(np.abs(emg_grid))))
    if aux.size != 0:
        print("aux min:", np.min(aux, axis=0).tolist())
        print("aux max:", np.max(aux, axis=0).tolist())

def plot_preview(emg_grid, aux, bad_mask_grid, time_s, grid=0, r=0, c=0, aux_col=0, overlay_bad=True):
    import matplotlib.pyplot as plt

    g = int(np.clip(grid, 0, 5))
    r = int(np.clip(r, 0, 7))
    c = int(np.clip(c, 0, 7))

    y = emg_grid[:, g, r, c]
    plt.figure()
    plt.plot(time_s, y)
    title = f"EMG (grid={g}, row={r}, col={c})"
    if overlay_bad and bad_mask_grid is not None and bad_mask_grid[g, r, c]:
        title += "  [BAD]"
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

    if aux.size != 0 and aux_col < aux.shape[1]:
        plt.figure()
        plt.plot(time_s, aux[:, aux_col])
        plt.title(f"AUX (col={aux_col})")
        plt.xlabel("Time (s)")
        plt.ylabel("AUX")
        plt.grid(True)
        plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to exported .npz")
    ap.add_argument("--plot", action="store_true", help="Show a quick plot preview")
    ap.add_argument("--grid", type=int, default=0, help="Grid index 0..5 for plotting")
    ap.add_argument("--row", type=int, default=0, help="Row index 0..7 for plotting (final grid coords)")
    ap.add_argument("--col", type=int, default=0, help="Col index 0..7 for plotting (final grid coords)")
    ap.add_argument("--aux-col", type=int, default=0, help="AUX column index for plotting")
    args = ap.parse_args()

    emg_grid, aux, bad_mask_grid, time_s, meta = load_export_npz(args.path)

    if time_s is None:
        # If missing, synthesize a simple index-based time vector
        time_s = np.arange(emg_grid.shape[0], dtype=float)

    summarize(emg_grid, aux, bad_mask_grid, time_s, meta)

    if args.plot:
        plot_preview(
            emg_grid, aux, bad_mask_grid, time_s,
            grid=args.grid, r=args.row, c=args.col, aux_col=args.aux_col
        )

if __name__ == "__main__":
    main()

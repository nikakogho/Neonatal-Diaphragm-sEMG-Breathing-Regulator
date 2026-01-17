import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_creation_utils import create_train_test_datasets_from_folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Folder containing .npz recordings")
    ap.add_argument("--test", default=None, help="Test recording name OR index (optional)")
    args = ap.parse_args()

    # parse test arg
    test_rec = None
    if args.test is not None:
        # try int first
        try:
            test_rec = int(args.test)
        except ValueError:
            test_rec = args.test

    train_ds, test_ds, info = create_train_test_datasets_from_folder(
        args.data,
        test_recording=test_rec,
        win_ms=200.0,
        delta_ms=100.0,
        step_ms=50.0,
        normalize=True,
        include_mask_channels=True,
    )

    print("\n=== DATASET INFO ===")
    for k, v in info.items():
        print(f"{k}: {v}")

    # 1) Basic shape checks (single item)
    X0, y0 = train_ds[0]
    print("\n=== SINGLE SAMPLE CHECK ===")
    print("X0 shape:", tuple(X0.shape), "expected (T, 12, 8, 8)")
    print("y0 shape:", tuple(y0.shape), "expected (2,)")

    T = X0.shape[0]
    assert X0.shape[1:] == (12, 8, 8), "X channel/shape mismatch"
    assert y0.shape == (2,), "y shape mismatch"

    # 2) Mask channel must be exactly 0/1 (NOT normalized!)
    mask = X0[:, 6:, :, :]  # last 6 channels
    uniq = torch.unique(mask)
    print("\n=== MASK CHANNEL CHECK ===")
    print("unique(mask values) =", uniq[:10], "(showing up to 10)")
    assert torch.all((mask == 0) | (mask == 1)), "Mask channels are not 0/1 (normalization bug)"

    # 3) Zeros at bad electrodes (where mask==0) in signal channels
    sig = X0[:, :6, :, :]
    zero_positions = (mask == 0)
    max_abs_at_bad = torch.max(torch.abs(sig[zero_positions]))
    print("\n=== ZEROING CHECK ===")
    print("max(|signal|) where mask==0 :", float(max_abs_at_bad))
    assert float(max_abs_at_bad) == 0.0, "Signal not zero at masked positions"

    # 4) Batch check: DataLoader and normalization sanity
    loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
    Xb, yb = next(iter(loader))
    print("\n=== BATCH CHECK ===")
    print("X batch:", tuple(Xb.shape), "expected (B, T, 12, 8, 8)")
    print("y batch:", tuple(yb.shape), "expected (B, 2)")

    sig_b = Xb[:, :, :6, :, :].reshape(-1)
    print("signal mean approx:", float(sig_b.mean()))
    print("signal std approx :", float(sig_b.std()))

    # mask still 0/1 in batch
    mask_b = Xb[:, :, 6:, :, :]
    assert torch.all((mask_b == 0) | (mask_b == 1)), "Mask channels not 0/1 in batch"

    print("\n✅ Sanity checks PASSED.")


if __name__ == "__main__":
    main()

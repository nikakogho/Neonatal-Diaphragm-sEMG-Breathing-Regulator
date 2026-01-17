import argparse
import os
import json
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset_creation_utils import create_train_test_datasets_from_folder
from model import EMG3DCNNRegressor


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_device(batch, device):
    X, y = batch
    return X.to(device), y.to(device)


@torch.no_grad()
def eval_epoch(model, loader, device, loss_fn, y_mean, y_std):
    """
    Returns:
      norm_mse: MSE on normalized y
      mae_phys: MAE in original units (de-normalized)
    """
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    n = 0

    y_mean = y_mean.to(device)
    y_std = y_std.to(device)

    for batch in loader:
        X, y = to_device(batch, device)
        pred = model(X)

        loss = loss_fn(pred, y)
        total_mse += float(loss.item()) * X.shape[0]

        # de-normalize for MAE in physical units
        pred_phys = pred * y_std + y_mean
        y_phys = y * y_std + y_mean
        mae = torch.mean(torch.abs(pred_phys - y_phys), dim=1)  # per-sample (averaged over 2 targets)
        total_mae += float(mae.sum().item())
        n += X.shape[0]

    return total_mse / max(1, n), total_mae / max(1, n)


def train_epoch(model, loader, device, loss_fn, opt):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        X, y = to_device(batch, device)
        pred = model(X)
        loss = loss_fn(pred, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total += float(loss.item()) * X.shape[0]
        n += X.shape[0]
    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Folder with .npz recordings")
    ap.add_argument("--test", default=None, help="Test recording name OR index (default: last file)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--win_ms", type=float, default=200.0)
    ap.add_argument("--delta_ms", type=float, default=100.0)
    ap.add_argument("--step_ms", type=float, default=50.0)

    ap.add_argument("--no_norm", action="store_true")
    ap.add_argument("--no_mask_channels", action="store_true")

    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.15)

    ap.add_argument("--patience", type=int, default=12, help="Early stopping patience on test loss")
    ap.add_argument("--overfit", type=int, default=0, help="If >0: overfit this many train samples (smoke test)")

    ap.add_argument("--no_scheduler", action="store_true", help="If set, do not use LR scheduler")

    ap.add_argument("--run_name", default=None)
    args = ap.parse_args()

    set_seed(args.seed)

    # parse test
    test_rec = None
    if args.test is not None:
        try:
            test_rec = int(args.test)
        except ValueError:
            test_rec = args.test

    normalize = not args.no_norm
    include_mask_channels = not args.no_mask_channels

    train_ds, test_ds, info = create_train_test_datasets_from_folder(
        args.data,
        test_recording=test_rec,
        win_ms=args.win_ms,
        delta_ms=args.delta_ms,
        step_ms=args.step_ms,
        normalize=normalize,
        include_mask_channels=include_mask_channels,
    )

    # optional overfit subset
    if args.overfit > 0:
        k = min(args.overfit, len(train_ds))
        train_ds = Subset(train_ds, list(range(k)))
        print(f"\n[OVERFIT MODE] Training on only {k} samples (should drive train loss near 0).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_ch = 6 if not include_mask_channels else 12

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, drop_last=False,
        num_workers=0, pin_memory=torch.cuda.is_available(), generator=g
    )
    train_eval_loader = DataLoader(
        train_ds, batch_size=max(64, args.batch), shuffle=False, drop_last=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=max(64, args.batch), shuffle=False, drop_last=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )

    model = EMG3DCNNRegressor(in_ch=in_ch, base=args.base, dropout=args.dropout).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    if args.no_scheduler:
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)

    # tensors for de-normalization metrics
    y_mean = torch.tensor(info["y_mean"], dtype=torch.float32)
    y_std = torch.tensor(info["y_std"], dtype=torch.float32)

    # run folder
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"emg_{ts}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("\n=== RUN INFO ===")
    print("device:", device)
    print("recordings:", info["recordings"])
    print("train recordings:", info["train_recording_names"])
    print("test recording:", info["test_recording_name"])
    print("win_samples:", info["win_samples"], "delta_samples:", info["delta_samples"], "step_samples:", info["step_samples"])
    print("X channels:", in_ch, "| normalize:", normalize)
    print("train windows:", info["n_train_windows"], "| test windows:", info["n_test_windows"])
    print("saving to:", run_dir)

    best = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_mse = train_epoch(model, train_loader, device, loss_fn, opt)
        train_mse_eval, train_mae_phys = eval_epoch(model, train_eval_loader, device, loss_fn, y_mean, y_std)
        test_mse, test_mae_phys = eval_epoch(model, test_loader, device, loss_fn, y_mean, y_std)
        if scheduler is not None:
            scheduler.step(test_mse)

        lr_now = opt.param_groups[0]["lr"]
        history.append({
            "epoch": epoch,
            "train_mse_norm": train_mse,
            "train_mse_norm_eval": train_mse_eval,
            "train_mae_phys": train_mae_phys,
            "test_mse_norm": test_mse,
            "test_mae_phys": test_mae_phys,
            "lr": lr_now,
        })

        print(
            f"Epoch {epoch:03d} | lr={lr_now:.2e} | "
            f"trainMSE(norm)={train_mse:.6f} | trainMSEeval(norm)={train_mse_eval:.6f} | trainMAE(phys)={train_mae_phys:.6f} | "
            f"testMSE(norm)={test_mse:.6f} | testMAE(phys)={test_mae_phys:.6f}"
        )

        # save best
        score = train_mse_eval if args.overfit > 0 else test_mse
        if score < best:
            best = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_test_mse_norm": float(test_mse),
                    "best_train_mse_norm_eval": float(train_mse_eval),
                    "best_score": float(best),
                    "args": vars(args),
                    "dataset_info": info,
                },
                os.path.join(run_dir, "best.pt"),
            )
            print("  ✅ saved best.pt")
        else:
            bad_epochs += 1

        # early stop (but not in overfit mode)
        if args.overfit <= 0 and bad_epochs >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs. Best epoch={best_epoch}, best testMSE(norm)={best:.6f}")
            break

    with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n=== DONE ===")
    if args.overfit > 0:
        print("best trainMSEeval(norm):", best, "at epoch", best_epoch)
    else:
        print("best testMSE(norm):", best, "at epoch", best_epoch)
    print("run dir:", run_dir)


if __name__ == "__main__":
    main()

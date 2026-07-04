"""Train the autoencoder on normal telemetry (the SMD train split).

Reconstruction-based detection rests on training with normal data only:
the model learns to reconstruct "normal", so at test time anything it
reconstructs poorly is suspect.
"""

import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .data import MinMaxScaler, load_machine
from .model import DenseAutoencoder


def main():
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    train_raw, _, _ = load_machine()
    scaler = MinMaxScaler().fit(train_raw)
    train = scaler.transform(train_raw).astype(np.float32)

    # chronological split: the last 10% is validation (a random split would
    # leak adjacent, nearly identical timesteps into validation)
    n_val = int(len(train) * config.VAL_FRACTION)
    train_x, val_x = train[:-n_val], train[-n_val:]

    model = DenseAutoencoder()
    opt = torch.optim.Adam(model.parameters(), lr=config.LR)
    loss_fn = nn.MSELoss()

    loader = DataLoader(TensorDataset(torch.from_numpy(train_x)),
                        batch_size=config.BATCH_SIZE, shuffle=True)
    val_t = torch.from_numpy(val_x)

    best_val, best_state, patience_left = float("inf"), None, config.PATIENCE
    for epoch in range(1, config.MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            opt.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(batch)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_t), val_t).item()
        print(f"epoch {epoch:3d}  train {epoch_loss / len(train_x):.6f}"
              f"  val {val_loss:.6f}")
        if val_loss < best_val - 1e-6:
            best_val, patience_left = val_loss, config.PATIENCE
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left == 0:
                print("early stopping")
                break

    model.load_state_dict(best_state)

    # Per-metric reconstruction-error statistics on normal data: the
    # reference distribution used to z-score errors into contribution scores.
    model.eval()
    with torch.no_grad():
        recon = model(torch.from_numpy(train)).numpy()
    feat_err = (recon - train) ** 2
    err_mean, err_std = feat_err.mean(axis=0), feat_err.std(axis=0) + 1e-8

    config.ARTIFACT_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), config.ARTIFACT_DIR / "autoencoder.pt")
    np.savez(config.ARTIFACT_DIR / "scaler.npz",
             min=scaler.min_, range=scaler.range_,
             err_mean=err_mean, err_std=err_std)
    (config.ARTIFACT_DIR / "train_meta.json").write_text(json.dumps({
        "machine": config.MACHINE,
        "best_val_mse": best_val,
        "epochs_ran": epoch,
    }, indent=2))
    print(f"saved model + scaler stats to {config.ARTIFACT_DIR}")


if __name__ == "__main__":
    main()

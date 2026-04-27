import torch
import numpy as np


def train(model, train_loader, val_loader, config, exp_path):
    device = config.DEVICE
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    best_val = float("inf")
    patience = config.EARLY_STOPPING_PATIENCE
    min_delta = config.EARLY_STOPPING_MIN_DELTA
    epochs_without_improvement = 0

    history = {"train": [], "val": [], "best_val": None, "stopped_epoch": None}

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(X)
            target = y.reshape(out.shape)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                target = y.reshape(out.shape)
                val_loss += criterion(out, target).item()

        val_loss /= len(val_loader)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        print(f"Epoch {epoch}: {train_loss:.4f} / {val_loss:.4f}")

        # Guardar mejor modelo y reiniciar paciencia si la validacion mejora.
        if val_loss < best_val - min_delta:
            best_val = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f"{exp_path}/models/best_model.pt")
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                history["stopped_epoch"] = epoch + 1
                print(
                    f"Early stopping en epoch {epoch + 1}. "
                    f"Mejor val_loss: {best_val:.4f}"
                )
                break

    history["best_val"] = best_val

    return history

def evaluate(model, loader, config, scaler):
    device = config.DEVICE
    model.eval()

    preds, trues = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            out = model(X).cpu().numpy()

            preds.extend(out)
            trues.extend(y.squeeze(1).numpy())

    preds = np.array(preds).reshape(-1, 1)
    trues = np.array(trues).reshape(-1, 1)

    preds_real = scaler.inverse_transform(preds)
    trues_real = scaler.inverse_transform(trues)

    errors = preds_real - trues_real
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    # Evita division por cero en MAPE cuando hay valores reales cercanos a 0.
    eps = 1e-8
    denominator = np.clip(np.abs(trues_real), eps, None)
    mape = np.mean(np.abs(errors) / denominator) * 100.0

    trues_flat = trues_real.ravel()
    preds_flat = preds_real.ravel()
    ss_res = np.sum((trues_flat - preds_flat) ** 2)
    ss_tot = np.sum((trues_flat - np.mean(trues_flat)) ** 2)
    if np.isclose(ss_tot, 0.0):
        r2 = 1.0 if np.isclose(ss_res, 0.0) else 0.0
    else:
        r2 = 1.0 - (ss_res / ss_tot)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2)}

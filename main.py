from core import Config, set_seed, build_model, train, evaluate
from core.data import load_data, preprocess, split, scale, create_sequences
from core.utils import create_experiment_folder, save_config, log_metrics

import torch
from torch.utils.data import DataLoader, TensorDataset

def run():
    config = Config()

    # Seed, para hacer reproducibles los experimentos
    set_seed(config.SEED)

    # Crear experimento
    exp_path = create_experiment_folder()
    save_config(config, exp_path)

    # Data
    df = load_data(config.DATA_PATH)
    df = preprocess(df, config)

    # Split(Entrenamiento, Validación, Test), Scale(Normalizamos utilizando el scaler MaxMinScaler) y Crear secuencias
    train_df, val_df, test_df = split(df, config)
    train_s, val_s, test_s, scaler= scale(train_df, val_df, test_df)

    X_train, y_train = create_sequences(train_s, config.WINDOW_SIZE, config.HORIZON)
    X_val, y_val = create_sequences(val_s, config.WINDOW_SIZE, config.HORIZON)
    X_test, y_test = create_sequences(test_s, config.WINDOW_SIZE, config.HORIZON)

    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train).float(),
        torch.tensor(y_train).float()
    ), batch_size=config.BATCH_SIZE, shuffle=False)

    val_loader = DataLoader(TensorDataset(
        torch.tensor(X_val).float(),
        torch.tensor(y_val).float()
    ), batch_size=config.BATCH_SIZE, shuffle=False)

    test_loader = DataLoader(TensorDataset(
        torch.tensor(X_test).float(),
        torch.tensor(y_test).float()
    ), batch_size=config.BATCH_SIZE, shuffle=False)

    # Modelo (la logica de construccion vive en models.build_model)
    model = build_model(config)

    # Train
    history = train(model, train_loader, val_loader, config, exp_path)

    # Load best model
    model.load_state_dict(torch.load(f"{exp_path}/models/best_model.pt"))

    # Evaluate
    metrics = evaluate(model, test_loader, config, scaler)

    # Save results
    log_metrics(exp_path, metrics, config)

    print("Final Metrics:", metrics)


if __name__ == "__main__":
    run()

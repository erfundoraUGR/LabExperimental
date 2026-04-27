"""
Optimización de hiperparámetros con Optuna basada en configuración YAML.
Permite optimizar cualquier combinación de dataset + modelo + parámetros.

Uso:
    python tune_optuna.py --config tune_config_transformer.yaml
    python tune_optuna.py --config tune_config_gru.yaml --n-trials 100
    python tune_optuna.py --config tune_config_lstm.yaml --model GRU

Archivos de configuración disponibles:
    - tune_config_transformer.yaml
    - tune_config_lstm.yaml
    - tune_config_gru.yaml
    - tune_config_mlp.yaml
    - tune_config_elman.yaml
"""

import argparse
from optuna_tuner import OptunaSimpleTuner


def parse_args():
    """Argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Optimización flexible de hiperparámetros con Optuna"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Ruta del archivo YAML de configuración",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Sobrescribir n_trials de la configuración",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Sobrescribir modelo de la configuración",
    )
    return parser.parse_args()


def main():
    """Función principal."""
    args = parse_args()

    tuner = OptunaSimpleTuner(
        config_path=args.config,
        override_n_trials=args.n_trials,
        override_model=args.model,
    )

    tuner.optimize()


if __name__ == "__main__":
    main()

"""
Optimization module - Sistema de optimización de hiperparámetros con Optuna.
Permite optimizar cualquier combinación de dataset + modelo + parámetros
basado en configuración YAML.

Uso:
    python optimization/tune_optuna.py --config optimization/configs/tune_config_transformer.yaml
"""

from .optuna_tuner import OptunaSimpleTuner

__all__ = ["OptunaSimpleTuner"]

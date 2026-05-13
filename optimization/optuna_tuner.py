"""
Tuner flexible basado en configuración YAML para optimización de parámetros.
Soporta cualquier combinación de dataset + modelo + parámetros.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

try:
    import optuna
except ImportError as exc:
    raise ImportError(
        "Optuna no está instalado. Ejecuta: pip install optuna"
    ) from exc

import sys
from pathlib import Path

# Agregar el directorio padre al path para importar desde core
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import Config, build_model, train, evaluate, load_data, preprocess, split, scale, create_sequences, set_seed


class OptunaSimpleTuner:
    """Tuner flexible que lee configuración de YAML."""

    def __init__(self, config_path: str, override_n_trials: int = None, override_model: str = None):
        """
        Args:
            config_path: Ruta al archivo YAML de configuración
            override_n_trials: Sobrescribir n_trials de la config
            override_model: Sobrescribir modelo de la config
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")

        # Cargar configuración YAML
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Sobrescrituras
        if override_n_trials is not None:
            self.config["n_trials"] = override_n_trials
        if override_model is not None:
            self.config["model"] = override_model

        # Validaciones
        self._validate_config()

        # Cargar configuración base del proyecto
        self.base_config = Config()
        self._resolve_device()

        # Setup de carpetas
        self._setup_output_dir()

        # Cachés
        self._loaders_cache = None

    def _validate_config(self):
        """Valida que la configuración tenga todos los campos necesarios."""
        required = ["dataset", "model", "n_trials", "parameters"]
        for field in required:
            if field not in self.config:
                raise ValueError(f"Campo requerido en YAML: {field}")

        if not Path(self.config["dataset"]).exists():
            raise FileNotFoundError(f"Dataset no encontrado: {self.config['dataset']}")

        valid_models = ["MLP", "LSTM", "GRU", "ELMAN", "TRANSFORMER"]
        if self.config["model"].upper() not in valid_models:
            raise ValueError(
                f"Modelo '{self.config['model']}' no válido. "
                f"Use: {', '.join(valid_models)}"
            )

    def _resolve_device(self):
        """Si CUDA no está disponible, usa CPU."""
        requested = str(self.base_config.DEVICE).lower()
        if requested == "cuda" and not torch.cuda.is_available():
            print("CUDA no disponible; se usará CPU.")
            self.base_config.DEVICE = "cpu"

    def _setup_output_dir(self):
        """Crea estructura de directorios de salida."""
        model_name = self.config["model"].upper()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("experiments") / "tuning" / f"{model_name}_{timestamp}"

        self.trials_dir = self.output_dir / "trials"
        self.reports_dir = self.output_dir / "reports"

        self.trials_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Guardar config usada
        with open(self.output_dir / "config.yaml", "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def _json_dump(self, path: Path, data: Dict):
        """Guarda JSON legible."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def _build_loaders(self):
        """Construye loaders respetando el flujo temporal."""
        if self._loaders_cache is not None:
            return self._loaders_cache

        df = load_data(self.config["dataset"])
        df = preprocess(df, self.base_config)

        train_df, val_df, test_df = split(df, self.base_config)
        train_s, val_s, test_s, scaler = scale(train_df, val_df, test_df)

        X_train, y_train = create_sequences(
            train_s, self.base_config.WINDOW_SIZE, self.base_config.HORIZON
        )
        X_val, y_val = create_sequences(
            val_s, self.base_config.WINDOW_SIZE, self.base_config.HORIZON
        )
        X_test, y_test = create_sequences(
            test_s, self.base_config.WINDOW_SIZE, self.base_config.HORIZON
        )

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train).float(),
                torch.tensor(y_train).float(),
            ),
            batch_size=self.base_config.BATCH_SIZE,
            shuffle=False,
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_val).float(),
                torch.tensor(y_val).float(),
            ),
            batch_size=self.base_config.BATCH_SIZE,
            shuffle=False,
        )
        test_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_test).float(),
                torch.tensor(y_test).float(),
            ),
            batch_size=self.base_config.BATCH_SIZE,
            shuffle=False,
        )

        self._loaders_cache = (train_loader, val_loader, test_loader, scaler)
        return self._loaders_cache

    def _build_search_space(self, trial) -> Dict[str, Any]:
        """Construye dinámicamente el espacio de búsqueda desde la config."""
        params = {}

        for param_name, param_config in self.config["parameters"].items():
            param_type = param_config.get("type")

            if param_type == "categorical":
                values = param_config.get("values")
                params[param_name] = trial.suggest_categorical(param_name, values)

            elif param_type == "int":
                min_val = param_config.get("min")
                max_val = param_config.get("max")
                params[param_name] = trial.suggest_int(param_name, min_val, max_val)

            elif param_type == "float":
                min_val = param_config.get("min")
                max_val = param_config.get("max")
                log = param_config.get("log", False)
                params[param_name] = trial.suggest_float(
                    param_name, min_val, max_val, log=log
                )

            elif param_type == "categorical_multi":
                # Parámetro dinámico: genera lista según num_layers
                num_layers_ref = param_config.get("num_layers_ref")
                if num_layers_ref and num_layers_ref in params:
                    num_layers = params[num_layers_ref]
                    values = param_config.get("values")
                    hidden_sizes = []
                    for i in range(num_layers):
                        hidden_sizes.append(
                            trial.suggest_categorical(
                                f"{param_name}_{i}", values
                            )
                        )
                    params[param_name] = hidden_sizes

            elif param_type == "int_multi":
                # Como categorical_multi pero con rango entero continuo
                num_layers_ref = param_config.get("num_layers_ref")
                if num_layers_ref and num_layers_ref in params:
                    num_layers = params[num_layers_ref]
                    min_val = param_config.get("min")
                    max_val = param_config.get("max")
                    hidden_sizes = []
                    for i in range(num_layers):
                        hidden_sizes.append(
                            trial.suggest_int(
                                f"{param_name}_{i}", min_val, max_val
                            )
                        )
                    params[param_name] = hidden_sizes

        return params

    def _apply_params_to_config(self, config: Config, params: Dict[str, Any]):
        """Aplica parámetros al config."""
        for key, value in params.items():
            setattr(config, key, value)

    def _create_trial_folder(
        self, trial_number: int, seed: int, stage: str = "search"
    ) -> Path:
        """Crea carpeta para un trial."""
        path = (
            self.trials_dir
            / stage
            / f"trial_{trial_number:04d}"
            / f"seed_{seed}"
        )
        (path / "models").mkdir(parents=True, exist_ok=True)
        (path / "results").mkdir(parents=True, exist_ok=True)
        (path / "logs").mkdir(parents=True, exist_ok=True)
        return path

    def _run_single_training(
        self, config: Config, exp_path: Path
    ) -> tuple:
        """Ejecuta entrenamiento y devuelve métricas."""
        train_loader, val_loader, test_loader, scaler = self._build_loaders()
        model = build_model(config)

        train(model, train_loader, val_loader, config, str(exp_path))
        best_model_path = exp_path / "models" / "best_model.pt"
        model.load_state_dict(
            torch.load(best_model_path, map_location=config.DEVICE)
        )

        val_metrics = evaluate(model, val_loader, config, scaler)
        test_metrics = evaluate(model, test_loader, config, scaler)

        return val_metrics, test_metrics

    def _evaluate_with_seeds(
        self,
        params: Dict[str, Any],
        seeds: List[int],
        trial_number: int,
        stage: str = "search",
        trial: optuna.Trial = None,
    ) -> Dict[str, Any]:
        """Evalúa parámetros con múltiples semillas."""
        objective_metric = self.base_config.TUNING_OBJECTIVE_METRIC
        alpha = self.base_config.TUNING_STABILITY_ALPHA

        val_scores = []
        per_seed = []

        for seed in seeds:
            config = Config()
            self._apply_params_to_config(config, params)
            config.SEED = seed
            config.EPOCHS = self.base_config.TUNING_EPOCHS
            config.MODEL_TYPE = self.config["model"].upper()
            config.DATA_PATH = self.config["dataset"]
            self._resolve_device()
            config.DEVICE = self.base_config.DEVICE

            set_seed(seed)
            exp_path = self._create_trial_folder(trial_number, seed, stage)

            try:
                val_metrics, test_metrics = self._run_single_training(
                    config, exp_path
                )

                val_metric = float(val_metrics[objective_metric])
                val_scores.append(val_metric)
                per_seed.append(
                    {
                        "seed": seed,
                        "val_metrics": val_metrics,
                        "test_metrics": test_metrics,
                    }
                )

                self._json_dump(
                    exp_path / "results" / "seed_metrics.json", per_seed[-1]
                )

                # Reporte incremental para pruning
                if trial is not None:
                    current_mean = float(np.mean(val_scores))
                    trial.report(current_mean, step=len(val_scores))
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

            except Exception as e:
                print(
                    f"Error en trial {trial_number}, seed {seed}: {str(e)}"
                )
                if trial is not None:
                    raise

        if not val_scores:
            return None

        mean_val = float(np.mean(val_scores))
        std_val = float(np.std(val_scores))
        robust_score = float(mean_val + alpha * std_val)

        return {
            "objective_metric": objective_metric,
            "val_mean": mean_val,
            "val_std": std_val,
            "robust_score": robust_score,
            "per_seed": per_seed,
        }

    def optimize(self) -> Dict[str, Any]:
        """Ejecuta la optimización completa."""
        n_trials = self.config.get("n_trials", 50)
        n_seeds = self.config.get("n_seeds", self.base_config.TUNING_SEEDS)
        sampler_name = self.config.get("sampler", "TPE")
        use_pruner = self.config.get("use_pruner", True)

        # Si n_seeds es int, crear lista
        if isinstance(n_seeds, int):
            n_seeds = list(range(42, 42 + n_seeds))

        print(f"\n{'='*60}")
        print(f"Optimizando {self.config['model'].upper()}")
        print(f"{'='*60}")
        print(f"Dataset: {self.config['dataset']}")
        print(f"Trials: {n_trials}")
        print(f"Seeds: {n_seeds}")
        print(f"Parámetros a optimizar: {list(self.config['parameters'].keys())}")
        print(f"{'='*60}\n")

        sampler = (
            optuna.samplers.RandomSampler(seed=self.base_config.SEED)
            if sampler_name == "RANDOM"
            else optuna.samplers.TPESampler(seed=self.base_config.SEED)
        )

        pruner = (
            optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)
            if use_pruner
            else optuna.pruners.NopPruner()
        )

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            study_name=f"tuning_{self.config['model'].upper()}",
        )

        def objective(trial: optuna.Trial):
            try:
                params = self._build_search_space(trial)

                seed_result = self._evaluate_with_seeds(
                    params=params,
                    seeds=n_seeds,
                    trial_number=trial.number,
                    stage="search",
                    trial=trial,
                )

                if seed_result is None:
                    return float("inf")

                trial.set_user_attr("full_params", params)
                trial.set_user_attr("val_mean", seed_result["val_mean"])
                trial.set_user_attr("val_std", seed_result["val_std"])

                return seed_result["robust_score"]

            except optuna.exceptions.TrialPruned:
                raise
            except Exception as exc:
                trial.set_user_attr("error", str(exc))
                return float("inf")

        study.optimize(objective, n_trials=n_trials)

        # Exportar trials
        trials_df = study.trials_dataframe()
        trials_csv = self.reports_dir / "trials.csv"
        trials_df.to_csv(trials_csv, index=False)

        # Mejor trial
        best_trial = study.best_trial
        best_payload = {
            "trial_number": best_trial.number,
            "score": best_trial.value,
            "params": best_trial.user_attrs.get("full_params", {}),
        }
        self._json_dump(self.reports_dir / "best_trial.json", best_payload)

        # Reevaluar top-k
        completed_trials = [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        completed_trials = sorted(
            completed_trials,
            key=lambda item: float("inf") if item.value is None else item.value,
        )

        top_k = self.base_config.TUNING_TOP_K
        top_k_trials = completed_trials[:top_k]

        final_results = []
        for rank, trial in enumerate(top_k_trials, start=1):
            params = trial.user_attrs.get("full_params", {})
            result = self._evaluate_with_seeds(
                params=params,
                seeds=self.base_config.TUNING_FINAL_SEEDS,
                trial_number=trial.number,
                stage="final",
            )

            if result is not None:
                final_results.append(
                    {
                        "rank": rank,
                        "trial_number": trial.number,
                        "search_score": trial.value,
                        "final_robust_score": result["robust_score"],
                        "final_val_mean": result["val_mean"],
                        "final_val_std": result["val_std"],
                        "params": params,
                    }
                )

        final_results = sorted(
            final_results, key=lambda item: item["final_robust_score"]
        )
        self._json_dump(
            self.reports_dir / "topk_final_reevaluation.json", final_results
        )

        if final_results:
            self._json_dump(
                self.reports_dir / "best_config_final.json", final_results[0]
            )

        # Resumen
        summary = {
            "model": self.config["model"],
            "dataset": self.config["dataset"],
            "n_trials": n_trials,
            "best_trial_search": best_payload,
            "best_trial_final": final_results[0] if final_results else None,
            "output_dir": str(self.output_dir),
        }
        self._json_dump(self.reports_dir / "summary.json", summary)

        self._print_summary(summary)

        return summary

    def _print_summary(self, summary: Dict):
        """Imprime resumen de resultados."""
        print(f"\n{'='*60}")
        print("RESULTADOS DE LA OPTIMIZACIÓN")
        print(f"{'='*60}")
        print(f"Modelo: {summary['model']}")
        print(f"Dataset: {summary['dataset']}")
        print(f"\nMejor score (búsqueda): {summary['best_trial_search']['score']:.6f}")

        if summary["best_trial_final"]:
            best = summary["best_trial_final"]
            print(f"Mejor score (reevaluado): {best['final_robust_score']:.6f}")
            print(f"\nMejores parámetros:")
            for key, value in best["params"].items():
                if isinstance(value, list):
                    print(f"  - {key}: {value}")
                elif isinstance(value, float):
                    print(f"  - {key}: {value:.6f}")
                else:
                    print(f"  - {key}: {value}")

        print(f"\nResultados guardados en: {summary['output_dir']}")
        print(f"{'='*60}\n")

import torch
import numpy as np
import random
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import json
from datetime import datetime
import os


def config_to_dict(config):
    # Config usa atributos de clase; __dict__ de la instancia puede estar vacio.
    return {
        key: getattr(config, key)
        for key in dir(config)
        if key.isupper() and not callable(getattr(config, key))
    }

def create_experiment_folder():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"experiments/{timestamp}"
    os.makedirs(path + "/models", exist_ok=True)
    os.makedirs(path + "/logs", exist_ok=True)
    os.makedirs(path + "/results", exist_ok=True)
    return path

def save_config(config, path):
    with open(f"{path}/config.json", "w") as f:
        json.dump(config_to_dict(config), f, indent=4)

def log_metrics(path, metrics, config=None):
    with open(f"{path}/results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    if config is not None:
        result_with_config = {
            #"metrics": metrics,
            "config": config_to_dict(config),
        }
        with open(f"{path}/results/results_with_config.json", "w") as f:
            json.dump(result_with_config, f, indent=4)

# Lab Experimental - Estructura del Proyecto

## 📁 Organización de Directorios

```
LabExperimental/
├── core/                           # Código original del proyecto
│   ├── __init__.py
│   ├── config.py                   # Configuración global
│   ├── models.py                   # Definiciones de modelos (MLP, LSTM, GRU, ELMAN, Transformer)
│   ├── train.py                    # Lógica de entrenamiento y evaluación
│   ├── data.py                     # Procesamiento y carga de datos
│   └── utils.py                    # Funciones utilitarias
├── optimization/                   # Sistema de optimización de parámetros
│   ├── __init__.py
│   ├── optuna_tuner.py            # Clase OptunaSimpleTuner
│   ├── tune_optuna.py             # Script de entrada para optimización
│   ├── configs/                    # Archivos de configuración YAML
│   │   ├── tune_config_transformer.yaml
│   │   ├── tune_config_lstm.yaml
│   │   ├── tune_config_gru.yaml
│   │   ├── tune_config_mlp.yaml
│   │   └── tune_config_elman.yaml
│   └── README.md                   # Documentación del sistema de optimización
├── main.py                         # Script principal para entrenar modelos
├── dataset/                        # Datasets
└── experiments/                    # Resultados de experimentos (creado automáticamente)
```

---

## 🎯 Uso

### 1. Entrenar un Modelo (usando el código original)
```bash
python main.py
```

### 2. Optimizar Parámetros de un Modelo (usando Optuna)
```bash
python optimization/tune_optuna.py --config optimization/configs/tune_config_transformer.yaml
```

### 3. Personalizar Búsqueda de Optimización
```bash
python optimization/tune_optuna.py --config optimization/configs/tune_config_gru.yaml --n-trials 100 --model LSTM
```

---

## 📦 Modulo `core/`

Contiene el código original del proyecto:
- **config.py**: Configuración global (seeds, rutas, hiperparámetros por defecto)
- **models.py**: Arquitecturas de redes neuronales
- **train.py**: Loop de entrenamiento con early stopping
- **data.py**: Carga, preprocesamiento y creación de secuencias
- **utils.py**: Funciones auxiliares

### Importar desde core
```python
from core import Config, build_model, train, evaluate
from core.data import load_data, preprocess, split, scale, create_sequences
```

---

## 🚀 Modulo `optimization/`

Sistema flexible para optimizar parámetros de hiperparámetros:
- **optuna_tuner.py**: Implementación de `OptunaSimpleTuner`
- **tune_optuna.py**: Interfaz CLI
- **configs/**: Archivos YAML con definiciones de búsqueda

### Características
- ✅ Configuración basada en YAML
- ✅ Soporta múltiples tipos de parámetros (categorical, int, float, categorical_multi)
- ✅ Múltiples semillas para robustez
- ✅ Reevaluación de top-k configuraciones
- ✅ MedianPruner para cortar trials malos

### Importar desde optimization
```python
from optimization import OptunaSimpleTuner
```

### Estructura de Configuración YAML
```yaml
dataset: "dataset/combined_data_2009_2019.csv"
model: "TRANSFORMER"
n_trials: 50
n_seeds: 3
sampler: "TPE"
use_pruner: true

parameters:
  TRANSFORMER_NUM_HEADS:
    type: "categorical"
    values: [2, 4, 8]
  LR:
    type: "float"
    min: 1e-4
    max: 3e-3
    log: true
```

---

## 📝 Documentación

- **core/**: Ver docstrings en cada archivo
- **optimization/**: Ver `optimization/README.md` para guía completa

---

## 🔄 Flujo de Trabajo Típico

### Opción 1: Entrenar con parámetros predefinidos
```bash
python main.py
```

### Opción 2: Encontrar mejores parámetros
```bash
# Paso 1: Optimizar con Optuna
python optimization/tune_optuna.py --config optimization/configs/tune_config_transformer.yaml

# Paso 2: Consultar mejores parámetros en:
# experiments/tuning/TRANSFORMER_*/best_config_final.json

# Paso 3: Copiar parámetros a core/config.py

# Paso 4: Entrenar con nuevos parámetros
python main.py
```

---

## 🗂️ Resultados de Experimentos

Los experimentos se guardan automáticamente en:

### Entrenamientos normales
```
experiments/YYYYMMDD_HHMMSS/
├── models/
├── results/
└── config.json
```

### Optimizaciones con Optuna
```
experiments/tuning/MODELO_YYYYMMDD_HHMMSS/
├── config.yaml
├── best_config_final.json
├── summary.json
├── trials.csv
└── trials/
    ├── search/
    └── final/
```

---

## 🔗 Dependencias

```
torch
numpy
pandas
scikit-learn
optuna (para optimization/)
pyyaml
```

Instalar:
```bash
pip install torch numpy pandas scikit-learn optuna pyyaml
```


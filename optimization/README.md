# Guía de Optimización de Parámetros con Optuna (Basada en YAML)

## Resumen

El nuevo sistema permite optimizar **cualquier combinación de dataset + modelo + parámetros** de forma flexible y simple usando archivos de configuración YAML.

## ¿Cómo Funciona?

### 1. Selecciona un archivo de configuración
Elige el archivo YAML para el modelo que quieras optimizar.

### 2. (Opcional) Personaliza los parámetros
Edita el archivo YAML para definir exactamente qué parámetros optimizar y sus rangos.

### 3. Ejecuta la optimización
```bash
python tune_optuna.py --config tune_config_MODELO.yaml
```

## Ejemplos de Uso

### Ejemplo 1: Optimizar TRANSFORMER
```bash
python tune_optuna.py --config tune_config_transformer.yaml
```
Esto optimizará el modelo TRANSFORMER con el dataset `combined_data_2009_2019.csv` usando los parámetros definidos en el archivo YAML.

### Ejemplo 2: Optimizar GRU con más trials
```bash
python tune_optuna.py --config tune_config_gru.yaml --n-trials 100
```
Ejecuta 100 trials en lugar de los 50 predeterminados.

### Ejemplo 3: Cambiar el modelo en tiempo de ejecución
```bash
python tune_optuna.py --config tune_config_transformer.yaml --model LSTM
```
Usa la configuración de TRANSFORMER pero cambia el modelo a LSTM.

### Ejemplo 4: Optimizar LSTM
```bash
python tune_optuna.py --config tune_config_lstm.yaml
```

### Ejemplo 5: Optimizar MLP con 200 trials
```bash
python tune_optuna.py --config tune_config_mlp.yaml --n-trials 200
```

## Archivos de Configuración Disponibles

| Archivo | Modelo | Parámetros Optimizables |
|---------|--------|--------------------------|
| `tune_config_transformer.yaml` | TRANSFORMER | NUM_HEADS, NUM_LAYERS, D_MODEL, DROPOUT, LR, etc. |
| `tune_config_lstm.yaml` | LSTM | NUM_LAYERS, HIDDEN_SIZES, DROPOUT, LR, etc. |
| `tune_config_gru.yaml` | GRU | NUM_LAYERS, HIDDEN_SIZES, DROPOUT, LR, etc. |
| `tune_config_mlp.yaml` | MLP | NUM_LAYERS, HIDDEN_SIZES, DROPOUT, LR, etc. |
| `tune_config_elman.yaml` | ELMAN | NUM_LAYERS, HIDDEN_SIZES, NONLINEARITY, DROPOUT, LR, etc. |

## Personalizar un Archivo de Configuración

### Estructura del YAML

```yaml
# Dataset a usar
dataset: "dataset/combined_data_2009_2019.csv"

# Modelo a optimizar
model: "TRANSFORMER"

# Cantidad de trials en la búsqueda principal
n_trials: 50

# Cantidad de semillas para robustez
n_seeds: 3

# Sampler: TPE (recomendado) o RANDOM
sampler: "TPE"

# Usar MedianPruner para cortar trials malos
use_pruner: true

# Parámetros a optimizar y sus rangos
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

### Tipos de Parámetros Soportados

#### 1. `categorical` - Valores discretos
```yaml
WINDOW_SIZE:
  type: "categorical"
  values: [72, 96, 144, 288]
```

#### 2. `int` - Rango de enteros
```yaml
TRANSFORMER_NUM_LAYERS:
  type: "int"
  min: 1
  max: 4
```

#### 3. `float` - Rango de flotantes
```yaml
LR:
  type: "float"
  min: 1e-4
  max: 3e-3
  log: true  # log=true: búsqueda en escala logarítmica
```

#### 4. `categorical_multi` - Lista dinámica por número de layers
```yaml
LSTM_HIDDEN_SIZES:
  type: "categorical_multi"
  values: [32, 64, 128, 256]
  num_layers_ref: "LSTM_NUM_LAYERS"  # Se genera una lista según num_layers
```

## Ejemplos de Personalización

### Caso 1: Reducir el espacio de búsqueda
Edita `tune_config_gru.yaml` para optimizar con valores reducidos:

```yaml
parameters:
  WINDOW_SIZE:
    type: "categorical"
    values: [144, 288]  # Solo 2 opciones en lugar de 4

  LR:
    type: "float"
    min: 1e-4
    max: 1e-3  # Rango más estrecho
    log: true

  GRU_NUM_LAYERS:
    type: "int"
    min: 1
    max: 2  # Solo 1 o 2 layers
```

### Caso 2: Expandir el espacio de búsqueda
Aumenta las opciones:

```yaml
parameters:
  MLP_HIDDEN_SIZES:
    type: "categorical_multi"
    values: [16, 32, 64, 128, 256, 512]  # Más opciones
    num_layers_ref: "MLP_NUM_LAYERS"

  LR:
    type: "float"
    min: 1e-5  # Más bajo
    max: 1e-2  # Más alto
    log: true
```

### Caso 3: Crear nueva configuración personalizada
Crea `tune_config_custom.yaml`:

```yaml
dataset: "dataset/combined_data_2009_2019.csv"
model: "TRANSFORMER"
n_trials: 100
n_seeds: 5
sampler: "TPE"
use_pruner: true

parameters:
  # Solo optimizar estos 4 parámetros críticos
  TRANSFORMER_NUM_HEADS:
    type: "categorical"
    values: [4, 8, 16]

  TRANSFORMER_NUM_LAYERS:
    type: "int"
    min: 2
    max: 6

  LR:
    type: "float"
    min: 1e-4
    max: 1e-2
    log: true

  TRANSFORMER_DROPOUT:
    type: "float"
    min: 0.0
    max: 0.4
```

Luego ejecuta:
```bash
python tune_optuna.py --config tune_config_custom.yaml
```

## Estructura de Salida

Después de ejecutar la optimización, encontrarás:

```
experiments/tuning/TRANSFORMER_20260427_120000/
├── config.yaml                      # Config usada
├── best_trial.json                  # Mejor trial de búsqueda
├── best_config_final.json           # Mejor config reevaluada
├── summary.json                     # Resumen de resultados
├── trials.csv                       # Todos los trials
├── topk_final_reevaluation.json     # Top-k reevaluados
└── trials/
    ├── search/
    │   └── trial_0000/seed_42/...
    └── final/
        └── trial_0001/seed_42/...
```

### Archivos Clave

- **best_config_final.json**: La mejor configuración encontrada (reevaluada con más semillas)
- **summary.json**: Resumen general con métricas
- **trials.csv**: Todos los trials ejecutados (útil para análisis)

## Flujo de Optimización

1. **Fase 1: Búsqueda Principal (n_trials)**
   - Ejecuta n_trials con el espacio de búsqueda definido
   - Cada trial se evalúa con n_seeds para robustez
   - Incluye MedianPruner para cortar trials malos (si use_pruner=true)

2. **Fase 2: Reevaluación de Top-K**
   - Toma los mejores trials
   - Los reevalúa con más semillas (TUNING_FINAL_SEEDS de config.py)
   - Valida que la configuración es estable

3. **Fase 3: Exportación**
   - Guarda todos los resultados
   - Identifica la mejor configuración general

## Cambiar Dataset

Para usar un dataset diferente, edita cualquier archivo YAML:

```yaml
# Antes:
dataset: "dataset/combined_data_2009_2019.csv"

# Después:
dataset: "dataset/tu_dataset.csv"
```

O crea un nuevo archivo `tune_config_custom.yaml` con tu dataset.

## Cambiar Modelo

Opción 1: Crear nuevo archivo YAML con el modelo deseado
```bash
cp tune_config_transformer.yaml tune_config_lstm.yaml
# Edita model: "LSTM"
python tune_optuna.py --config tune_config_lstm.yaml
```

Opción 2: Sobrescribir en línea de comandos
```bash
python tune_optuna.py --config tune_config_transformer.yaml --model GRU
```

## Consejos de Optimización

### Para búsquedas rápidas:
```yaml
n_trials: 20       # Menos trials
n_seeds: 1         # Una sola semilla
use_pruner: true   # Corta trials malos rápidamente
```

### Para búsquedas exhaustivas:
```yaml
n_trials: 200      # Muchos trials
n_seeds: 5         # Más semillas para robustez
sampler: "TPE"     # TPE es más inteligente que RANDOM
```

### Para debugging:
```yaml
n_trials: 5        # Solo 5 para verificar que funciona
n_seeds: 1         # Una semilla
use_pruner: false  # Ver todos los trials completos
```

## Comandos Rápidos

```bash
# Optimizar cada modelo
python tune_optuna.py --config tune_config_transformer.yaml
python tune_optuna.py --config tune_config_lstm.yaml
python tune_optuna.py --config tune_config_gru.yaml
python tune_optuna.py --config tune_config_mlp.yaml
python tune_optuna.py --config tune_config_elman.yaml

# Búsqueda rápida
python tune_optuna.py --config tune_config_transformer.yaml --n-trials 10

# Búsqueda exhaustiva
python tune_optuna.py --config tune_config_gru.yaml --n-trials 300

# Cambiar modelo dinámicamente
python tune_optuna.py --config tune_config_transformer.yaml --model LSTM
```

## Reproducibilidad

La configuración usada se guarda automáticamente en:
```
experiments/tuning/MODEL_timestamp/config.yaml
```

Para reproducir la optimización exacta:
```bash
python tune_optuna.py --config experiments/tuning/TRANSFORMER_20260427_120000/config.yaml
```

## Preguntas Frecuentes

**P: ¿Cómo cambio qué parámetros optimizar?**
R: Edita el archivo YAML. Agrega o quita parámetros en la sección `parameters`.

**P: ¿Cómo hago una búsqueda más amplia?**
R: Aumenta `n_trials` y expande los rangos en el YAML.

**P: ¿Cómo hago una búsqueda más específica?**
R: Reduce `n_trials`, limita los rangos y elimina parámetros innecesarios.

**P: ¿Puedo usar un dataset diferente?**
R: Sí, edita `dataset` en el YAML.

**P: ¿Puedo optimizar múltiples modelos?**
R: Ejecuta cada uno independientemente o crea un script bash que llame a tune_optuna.py múltiples veces.

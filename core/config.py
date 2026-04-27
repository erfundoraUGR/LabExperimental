class Config:
    #Hiperparámetros
    
    SEED = 42

    DATA_PATH = "dataset/combined_data_2009_2019.csv"
    TARGET = "dem"

    RESAMPLE = "10min"

    TRAIN_SIZE = 0.7
    VAL_SIZE = 0.20
    TEST_SIZE = 0.10


    WINDOW_SIZE = 144
    HORIZON = 1

    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 0.0002844
    
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_MIN_DELTA = 1e-6

    MODEL_TYPE = "MLP"  # MLP | LSTM | GRU | ELMAN | TRANSFORMER

    # Arquitectura MLP configurable (antes estaba fija en models.py)
    MLP_HIDDEN_SIZES = [77, 57]
    MLP_DROPOUT = 0.0

    # Puedes definir una neurona distinta por capa LSTM, por ejemplo [128, 64, 32]
    LSTM_HIDDEN_SIZES = [64]
    LSTM_HIDDEN_SIZE = 64
    LSTM_NUM_LAYERS = 1
    LSTM_DROPOUT = 0.0

    # Puedes definir una neurona distinta por capa GRU, por ejemplo [128, 64, 32]
    GRU_HIDDEN_SIZES = [64]
    GRU_HIDDEN_SIZE = 64
    GRU_NUM_LAYERS = 1
    GRU_DROPOUT = 0.0

    # Puedes definir una neurona distinta por capa Elman, por ejemplo [128, 64, 32]
    ELMAN_HIDDEN_SIZES = [64]
    ELMAN_HIDDEN_SIZE = 64
    ELMAN_NUM_LAYERS = 1
    ELMAN_DROPOUT = 0.0
    ELMAN_NONLINEARITY = "tanh"  # "tanh" o "relu"

    # Parámetros Transformer
    TRANSFORMER_HIDDEN_SIZES = [64]  # Primera capa es d_model, luego capas lineales opcionales
    TRANSFORMER_HIDDEN_SIZE = 64     # (deprecated: usar TRANSFORMER_HIDDEN_SIZES)
    TRANSFORMER_NUM_HEADS = 4
    TRANSFORMER_NUM_LAYERS = 2
    TRANSFORMER_DROPOUT = 0.0


    # -----------------------------
    # Parametros para optimizacion
    # -----------------------------
    
    # Fase principal de optimizacion (TPE/Bayesian).
    TUNING_N_TRIALS = 40
    # Semillas para robustez durante cada trial.
    TUNING_SEEDS = [42, 123, 999]
    # Semillas para re-evaluar los mejores trials.
    TUNING_FINAL_SEEDS = [7, 21, 42, 123, 999]
    # Numero de mejores trials a reevaluar al final.
    TUNING_TOP_K = 5
    # Penalizacion de inestabilidad: score = media + alpha * std.
    TUNING_STABILITY_ALPHA = 0.25
    # Nombre de metrica para comparar trials.
    TUNING_OBJECTIVE_METRIC = "RMSE"
    # Epochs usados durante el tuning (con early stopping).
    TUNING_EPOCHS = 60  

    DEVICE = "cuda"  # o "cpu"

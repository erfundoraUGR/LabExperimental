class Config:
    SEED = 42

    DATA_PATH = "dataset/combined_data_2009_2019.csv"
    TARGET = "dem"
    RESAMPLE = "10min"

    # Split exacto del TFM: 70% train / 30% test (cronológico, sin shuffle).
    # Validación = último 30% del train (VAL_RATIO_OF_TRAIN=0.30).
    # El scaler se ajusta sobre el 70% completo del train (no solo sobre el subtrain).
    TRAIN_SIZE = 0.70
    TEST_SIZE  = 0.30
    VAL_RATIO_OF_TRAIN = 0.30   # fracción DEL TRAIN usada como validación

    # REE dataset: 24h lag = 144 steps, 48h lag = 288 steps (10-min intervals)
    WINDOW_SIZE = 144
    HORIZON = 1

    EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_MIN_DELTA = 1e-6

    # =========================================================================
    # Active model — set MODEL_TYPE, then update LR and BATCH_SIZE to match.
    # Source: TFM Javier Jiménez Arenas (UGR 2025), Tables 6.3 (24h) / 6.4 (48h)
    #
    #  Model        24h (WINDOW=144)              48h (WINDOW=288)
    #  MLP          LR=0.0002844  BS=32           LR=0.0003431  BS=32
    #  ELMAN        LR=0.0001589  BS=64           LR=0.0002024  BS=32
    #  LSTM         LR=0.0017554  BS=32           LR=0.0009840  BS=32
    #  GRU          LR=0.0008212  BS=16           LR=0.0008951  BS=16
    #  TRANSFORMER  LR=0.0005801  BS=32           LR=0.0001083  BS=16
    # =========================================================================
    MODEL_TYPE = "MLP"  # MLP | LSTM | GRU | ELMAN | TRANSFORMER

    LR = 0.0002844   # MLP 24h optimal — update when changing MODEL_TYPE
    BATCH_SIZE = 32  # MLP 24h optimal — update when changing MODEL_TYPE

    # --- MLP — TFM Table 6.3 (24h) / Table 6.4 (48h) ---
    MLP_HIDDEN_SIZES = [77, 57]   # 24h; 48h: [81, 32]
    MLP_DROPOUT = 0.0

    # --- LSTM — TFM Table 6.3 (24h) / Table 6.4 (48h) ---
    LSTM_HIDDEN_SIZES = [86, 16, 2]   # 24h; 48h: [86, 18]
    LSTM_HIDDEN_SIZE = 86
    LSTM_NUM_LAYERS = 3
    LSTM_DROPOUT = 0.0

    # --- GRU — TFM Table 6.3 (24h) / Table 6.4 (48h) ---
    GRU_HIDDEN_SIZES = [82, 46]   # 24h; 48h: [65, 39]
    GRU_HIDDEN_SIZE = 82
    GRU_NUM_LAYERS = 2
    GRU_DROPOUT = 0.0

    # --- ELMAN — TFM Table 6.3 (24h) / Table 6.4 (48h) ---
    ELMAN_HIDDEN_SIZES = [97, 44, 48]   # 24h; 48h: [19, 100]
    ELMAN_HIDDEN_SIZE = 97
    ELMAN_NUM_LAYERS = 3
    ELMAN_DROPOUT = 0.0
    ELMAN_NONLINEARITY = "tanh"

    # --- TRANSFORMER — TFM Table 6.3 (24h) / Table 6.4 (48h) ---
    # HIDDEN_SIZES[0] = d_model (embedding dim, must be divisible by NUM_HEADS)
    # HIDDEN_SIZES[1:] = FC layers after the encoder
    # 24h: emb=50 adjusted to 52 (nearest multiple of 4), FC=[71,100,85]
    # 48h: emb=92 (divisible by 2), FC=[47,75] → [92,47,75], NUM_HEADS=2, BS=16
    TRANSFORMER_HIDDEN_SIZES = [52, 71, 100, 85]   # 24h; 48h: [92, 47, 75]
    TRANSFORMER_HIDDEN_SIZE = 52
    TRANSFORMER_NUM_HEADS = 4                        # 24h; 48h: 2
    TRANSFORMER_NUM_LAYERS = 1
    TRANSFORMER_DROPOUT = 0.0

    # -------------------------------------------------------------------------
    # Optimization
    # -------------------------------------------------------------------------
    TUNING_N_TRIALS = 40
    TUNING_SEEDS = [42, 123, 999]
    TUNING_FINAL_SEEDS = [7, 21, 42, 123, 999]
    TUNING_TOP_K = 5
    TUNING_STABILITY_ALPHA = 0.25
    TUNING_OBJECTIVE_METRIC = "RMSE"
    TUNING_EPOCHS = 60

    DEVICE = "cuda"  # o "cpu"

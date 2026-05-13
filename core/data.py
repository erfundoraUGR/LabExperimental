import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(path):
    # Hay que tener en cuenta acá la estructura del dataset, separador, formato de fecha, tipos de datos de atributos, etc.
    df = pd.read_csv(path)

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["dem"] = pd.to_numeric(df["dem"], errors="coerce")

    df = df.dropna(subset=["ts", "dem"])
    df = df.set_index("ts").sort_index() 

    return df

def preprocess(df, config):
    # Nos quedamos solo con la columna objetivo para el modelado "dem"
    df= df[[config.TARGET]]     

    # Resampleamos a la frecuencia deseada (config.RESAMPLE) y nos quedamos solo con la columna objetivo (config.TARGET)
    df = df.resample(config.RESAMPLE).mean()   

    return df

def split(df, config):
    """Divide el dataframe en train, val y test de forma cronológica.

    Si config tiene VAL_RATIO_OF_TRAIN (esquema TFM):
      - test  = último TRAIN_SIZE*TEST_SIZE  → aquí TEST_SIZE = 1 - TRAIN_SIZE
      - val   = último VAL_RATIO_OF_TRAIN del bloque train
      - train = resto inicial del bloque train

    Si no, usa el esquema clásico TRAIN_SIZE / VAL_SIZE / TEST_SIZE.
    """
    n = len(df)
    if hasattr(config, 'VAL_RATIO_OF_TRAIN'):
        train_end = int(n * config.TRAIN_SIZE)
        full_train = df[:train_end]
        test_df   = df[train_end:]
        val_start = int(len(full_train) * (1 - config.VAL_RATIO_OF_TRAIN))
        return full_train[:val_start], full_train[val_start:], test_df
    else:
        train_end = int(n * config.TRAIN_SIZE)
        val_end   = int(n * (config.TRAIN_SIZE + config.VAL_SIZE))
        return df[:train_end], df[train_end:val_end], df[val_end:]

def scale(train, val, test, fit_on_train_and_val=True):
    """Normaliza MinMax [0,1].

    fit_on_train_and_val=True  (esquema TFM): ajusta el scaler sobre train+val
      combinados, igual que el TFM que ajusta sobre el 70% completo del dataset.
    fit_on_train_and_val=False (esquema clásico): ajusta solo sobre train.
    """
    scaler = MinMaxScaler()
    if fit_on_train_and_val:
        combined = np.concatenate([train, val], axis=0)
        scaler.fit(combined)
    else:
        scaler.fit(train)

    train_s = scaler.transform(train)
    val_s   = scaler.transform(val)
    test_s  = scaler.transform(test)

    return train_s, val_s, test_s, scaler

def create_sequences(data, window, horizon):
    X, y = [], []

    for i in range(len(data) - window - horizon):
        X.append(data[i:i+window])
        y.append(data[i+window:i+window+horizon])

    return np.array(X), np.array(y)

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
    n = len(df)
    train_end = int(n * config.TRAIN_SIZE)
    val_end = int(n * (config.TRAIN_SIZE + config.VAL_SIZE))

    return df[:train_end], df[train_end:val_end], df[val_end:]

def scale(train, val, test):
    scaler = MinMaxScaler()
    train_s = scaler.fit_transform(train)
    val_s = scaler.transform(val)
    test_s = scaler.transform(test)

    return train_s, val_s, test_s, scaler

def create_sequences(data, window, horizon):
    X, y = [], []

    for i in range(len(data) - window - horizon):
        X.append(data[i:i+window])
        y.append(data[i+window:i+window+horizon])

    return np.array(X), np.array(y)

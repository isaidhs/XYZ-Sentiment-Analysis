import pandas as pd
import pickle

def load_data(file_path):
    """
    Load data dari CSV dan filter berdasarkan tahun.
    """
    df = pd.read_csv(file_path)
    df['at'] = pd.to_datetime(df['at'])
    return df[df['at'].dt.year >= 2023]

def save_to_csv(df, file_path):
    """
    Menyimpan DataFrame ke file CSV.
    """
    df.to_csv(file_path, index=False)

def save_model(obj, file_path):
    """
    Menyimpan model atau objek lain menggunakan pickle.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def load_model(file_path):
    """
    Memuat model atau objek lain menggunakan pickle.
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)
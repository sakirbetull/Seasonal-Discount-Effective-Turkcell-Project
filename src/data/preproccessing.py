import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy.stats.mstats import winsorize

def load_data(filepath):
    """
    Veriyi yükler ve DataFrame olarak döndürür.
    """
    return pd.read_csv(filepath)

def feature_engineering(df):
    """
    Özellik mühendisliği işlemlerini gerçekleştirir.
    """
    # Toplam fiyat hesaplama
    df['total_price'] = ((1 - df['discount']) * (df['unit_price'] * df['quantity']))

    # Tarih işlemleri
    df["order_date"] = pd.to_datetime(df["order_date"])
    df['year'] = df['order_date'].dt.year
    df['yearquarter'] = df['year'].astype(str) + ' Q' + df['order_date'].dt.quarter.astype(str)

    # Label oluşturma
    df['label'] = np.where(df['quantity'] - df['expected_quantity'] > 0, 1, 0)

    return df

def handle_outliers(df):
    """
    Aykırı değerleri winsorize yöntemiyle manipüle eder.

    """
    df['unit_price_winsorized'] = winsorize(df['unit_price'].values, limits=[0.05, 0.05])
    df['total_price_winsorized'] = winsorize(df['total_price'].values, limits=[0.05, 0.05])
    return df
# logaritmik dönüşüm sağa çarpık dağılımı düzeltir(sağa çarpık ve bağımsız değişkenleri düşün quantity değil mesela )
def drop_unnecessary_columns(df):
    """
    Gereksiz sütunları kaldırır.
    """
    columns_to_drop = ['order_date', 'customer_id', 'product_id', 'category_id', 
                       'unit_price', 'total_price', 'year', 'expected_quantity']
    return df.drop(columns=columns_to_drop)

def save_data(df, output_filepath):
    """
    İşlenmiş veriyi CSV dosyasına kaydeder.
    """
    df.to_csv(output_filepath, index=False)
    print(f"İşlenmiş veri {output_filepath} dosyasına kaydedildi.")

if __name__ == "__main__":
    # Get the project root directory
    current_dir = Path(__file__).parent.parent.parent
    
    # Define file paths
    input_filepath = current_dir / "data" / "processed" / "tabledf_predictedquantity.csv"
    output_filepath = current_dir / "data" / "processed" / "preprocessed_data.csv"
    
    # Check if input file exists
    if not input_filepath.exists():
        raise FileNotFoundError(f"Input file not found at {input_filepath}. Please ensure the data file exists in the correct location.")
    
    # Create output directory if it doesn't exist
    os.makedirs(current_dir / "data" / "processed", exist_ok=True)
    
    # Veri yükleme
    df = load_data(str(input_filepath))

    # Özellik mühendisliği
    df = feature_engineering(df)

    # Aykırı değerleri işleme
    df = handle_outliers(df)

    # Gereksiz sütunları kaldırma
    df = drop_unnecessary_columns(df)

    # İşlenmiş veriyi kaydetme
    save_data(df, str(output_filepath))

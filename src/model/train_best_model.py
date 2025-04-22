import pandas as pd
import numpy as np
import os
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_and_preprocess_data(filepath):
    """
    Veriyi yükler, encode eder ve eğitim/test setlerini oluşturur.
    """
    df = pd.read_csv(filepath)

    # Label encoding
    df['city'] = df['city'].astype('category').cat.codes
    df['product_name'] = df['product_name'].astype('category').cat.codes

    # One-hot encoding
    df = pd.get_dummies(df, columns=['category_name', 'yearquarter'], drop_first=True)

    # Özellik ve hedef değişkenleri ayırma
    X = df.drop(columns=['label'])
    y = df['label']

    # Eğitim ve test setlerini ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Min-Max Scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def train_xgboost(X_train, y_train):
    """
    XGBoost modelini eğitir ve en iyi modeli döndürür.
    """
    xgb_params = {
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=xgb_params,
        n_iter=50,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    return best_model

def save_model(model, filepath):
    """
    Eğitilmiş modeli belirtilen dosya yoluna kaydeder.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    joblib.dump(model, filepath)
    print(f"Model başarıyla {filepath} dosyasına kaydedildi.")

if __name__ == "__main__":
    # Get the project root directory
    current_dir = Path(__file__).parent.parent.parent
    
    # Define file paths
    input_filepath = current_dir / "data" / "processed" / "preprocessed_data.csv"
    model_filepath = current_dir / "models" / "salesprediction_xgboost_model.pkl"
    
    # Check if input file exists
    if not input_filepath.exists():
        raise FileNotFoundError(f"Input file not found at {input_filepath}. Please ensure the data file exists in the correct location.")
    
    # Create models directory if it doesn't exist
    os.makedirs(current_dir / "models", exist_ok=True)
    
    # Veri yükleme ve ön işleme
    X_train, X_test, y_train, y_test = load_and_preprocess_data(str(input_filepath))

    # Model eğitimi
    best_xgb = train_xgboost(X_train, y_train)

    # Modeli kaydetme
    save_model(best_xgb, str(model_filepath))
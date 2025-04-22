import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

def preprocess_data(filepath):
    """
    Veriyi işler ve eğitim/test setlerini hazırlar.
    """
    df = pd.read_csv(filepath)

    # Feature Engineering
    df['total_price'] = ((1 - df['discount']) * (df['unit_price'] * df['quantity']))
    df["order_date"] = pd.to_datetime(df["order_date"])
    df['yearquarter'] = df['order_date'].dt.year.astype(str) + ' Q' + df['order_date'].dt.quarter.astype(str)

    # Aykırı değerleri winsorize etme
    df['unit_price_winsorized'] = winsorize(df['unit_price'].values, limits=[0.05, 0.05])
    df['total_price_winsorized'] = winsorize(df['total_price'].values, limits=[0.05, 0.05])

    # Gereksiz sütunları kaldırma
    df = df.drop(columns=['order_date', 'customer_id', 'product_id', 'category_id', 'unit_price', 'total_price'])

    # Encoding
    le = LabelEncoder()
    df['city'] = le.fit_transform(df['city'])
    df['product_name'] = le.fit_transform(df['product_name'])
    df = pd.get_dummies(df, columns=['category_name', 'yearquarter'], drop_first=True)

    # Eğitim ve test setlerini ayırma
    df_train = df[df['discount'] == 0].copy()
    df_test = df[df['discount'] != 0].copy()

    X_train = df_train.drop(columns=['quantity', 'discount'])
    y_train = df_train['quantity']

    X_test = df_test.drop(columns=['quantity', 'discount'])
    y_test = df_test['quantity']

    return X_train, y_train, X_test, y_test, df

def train_xgboost(X_train, y_train):
    """
    Random Forest modelini de denedik ama XGBoost daha iyi sonuç verdi.
    XGBoost modelini eğitir ve en iyi modeli döndürür.
    """
    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    xgb = XGBRegressor(random_state=42)
    xgb_model = GridSearchCV(estimator=xgb, param_grid=xgb_params, cv=3, scoring='r2', n_jobs=-1)
    xgb_model.fit(X_train, y_train)

    best_xgb = XGBRegressor(**xgb_model.best_params_, random_state=42)
    best_xgb.fit(X_train, y_train)  # En iyi parametrelerle yeniden eğit

    return best_xgb

def evaluate_model(model, X_test, y_test):
    """
    Modeli değerlendirir ve performans metriklerini döndürür.
    """
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return {"R2 Score": r2, "RMSE": rmse}, y_pred

def save_feature_importance(model, X_train, filename):
    """
    Özellik önemini görsel olarak kaydeder.
    """
    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    feature_importance.plot(kind="bar", x="Feature", y="Importance", figsize=(12, 6), title="Feature Importance")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_results_to_csv(original_df, processed_df, y_pred_test, y_pred_train, output_filepath):
    """
    Tahmin edilen değerleri orijinal tabloya ekler ve CSV olarak kaydeder.
    """
    # Orijinal tabloya expected_quantity sütununu ekle
    original_df['expected_quantity'] = np.nan

    # İndirimli ürünler için tahmin edilen değerleri ekle
    original_df.loc[processed_df['discount'] != 0, 'expected_quantity'] = y_pred_test

    # İndirimsiz ürünler için tahmin edilen değerleri ekle
    original_df.loc[processed_df['discount'] == 0, 'expected_quantity'] = y_pred_train

    # Sonuçları kaydet
    original_df.to_csv(output_filepath, index=False)
    print(f"Tahmin edilen değerler {output_filepath} dosyasına kaydedildi.")

if __name__ == "__main__":
    # Get the project root directory
    current_dir = Path(__file__).parent.parent.parent
    
    # Define file paths
    input_filepath = current_dir / "data" / "raw" / "tabledf.csv"
    output_filepath = current_dir / "data" / "processed" / "tabledf_predictedquantity.csv"
    
    # Check if input file exists
    if not input_filepath.exists():
        raise FileNotFoundError(f"Input file not found at {input_filepath}. Please ensure the data file exists in the correct location.")
    
    # Create output directory if it doesn't exist
    os.makedirs(current_dir / "data" / "processed", exist_ok=True)
    
    # Orijinal tabloyu oku
    original_df = pd.read_csv(str(input_filepath))

    # Veri işleme
    X_train, y_train, X_test, y_test, processed_df = preprocess_data(str(input_filepath))

    # Model eğitimi
    best_xgb = train_xgboost(X_train, y_train)

    # Model değerlendirme (indirimli ürünler için)
    metrics_test, y_pred_test = evaluate_model(best_xgb, X_test, y_test)

    # İndirimsiz ürünler için tahmin yap
    y_pred_train = best_xgb.predict(X_train)

    # Performans sonuçlarını yazdır
    print("\n--- Model Performans Sonuçları ---")
    print(f"R2 Score (Test): {metrics_test['R2 Score']:.4f}")
    print(f"RMSE (Test): {metrics_test['RMSE']:.4f}")

    # Create reports directory if it doesn't exist
    os.makedirs(current_dir / "reports" / "figures", exist_ok=True)
    
    # Feature importance görselini kaydet
    save_feature_importance(best_xgb, X_train, str(current_dir / "reports" / "figures" / "feature_importance.png"))

    # Tahmin edilen değerleri orijinal tabloya ekle ve kaydet
    save_results_to_csv(original_df, processed_df, y_pred_test, y_pred_train, str(output_filepath))
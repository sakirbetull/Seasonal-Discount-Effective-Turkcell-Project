import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight

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

    # Min-Max Scaling (KNN ve Logistic Regression için)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

def train_and_evaluate_model(model, params, X_train, y_train, X_test, y_test, scaled=False):
    """
    Modeli eğitir, en iyi parametreleri bulur ve performans metriklerini döndürür.
    """
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=50,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0,
        random_state=42
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": type(model).__name__,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "Best Model": best_model
    }

    return metrics, y_pred, y_proba

def plot_roc_curves(models, X_test, y_test):
    """
    Tüm modellerin ROC eğrilerini tek bir grafikte çizer.
    """
    # Create figures directory if it doesn't exist
    os.makedirs("reports/figures", exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    for model_info in models:
        model = model_info["Best Model"]
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{model_info['Model']} (AUC = {model_info['ROC-AUC']:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.savefig("reports/figures/roc_curves.png")
    plt.close()

def plot_feature_importance(model, feature_names):
    """
    En iyi modelin feature importance grafiğini çizer.
    """
    # Create figures directory if it doesn't exist
    os.makedirs("reports/figures", exist_ok=True)
    
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importance)), importance[sorted_idx], align="center")
    plt.xticks(range(len(importance)), np.array(feature_names)[sorted_idx], rotation=90)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("reports/figures/feature_importance.png")
    plt.close()

def plot_confusion_matrix(y_test, y_pred):
    """
    Confusion matrix'i çizer.
    """
    # Create figures directory if it doesn't exist
    os.makedirs("reports/figures", exist_ok=True)
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("reports/figures/confusion_matrix.png")
    plt.close()

if __name__ == "__main__":
    # Veri yükleme ve ön işleme
    # Use relative path instead of hardcoded path
    current_dir = Path(__file__).parent.parent
    filepath = current_dir / "data" / "processed" / "preprocessed_data.csv"
    
    # Check if file exists
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found at {filepath}. Please ensure the data file exists in the correct location.")
        
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_preprocess_data(str(filepath))

    # Modeller ve parametreler
    models = [
        (DecisionTreeClassifier(random_state=42), {
            "max_depth": [3, 5, 8, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 5, 10]
        }),
        (KNeighborsClassifier(), {
            "n_neighbors": np.arange(1, 50)
        }),
        (LogisticRegression(random_state=42), {
            "penalty": ['l1', 'l2'],
            "C": [0.1, 1],
            "solver": ['liblinear'],
            "max_iter": [100]
        }),
        (RandomForestClassifier(random_state=42, class_weight='balanced'), {
            "n_estimators": [100, 200, 500],
            "max_features": ['sqrt', 'log2'],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }),
        (XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), {
            "n_estimators": [100, 200, 500],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        })
    ]

    results = []
    for model, params in models:
        scaled = isinstance(model, (KNeighborsClassifier, LogisticRegression))
        X_train_input = X_train_scaled if scaled else X_train
        X_test_input = X_test_scaled if scaled else X_test
        metrics, y_pred, y_proba = train_and_evaluate_model(model, params, X_train_input, y_train, X_test_input, y_test)
        results.append(metrics)

    # Performans tablosu
    results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
    print("\n--- Model Performans Sonuçları ---")
    print(results_df)

    # ROC eğrilerini çiz
    plot_roc_curves(results, X_test, y_test)

    # En iyi modelin feature importance ve confusion matrix'i
    best_model_info = results[0]
    best_model = best_model_info["Best Model"]
    if hasattr(best_model, "feature_importances_"):
        plot_feature_importance(best_model, X_train.columns)
    plot_confusion_matrix(y_test, best_model.predict(X_test))
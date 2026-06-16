"""
train_model.py
================
Script untuk melatih model klasifikasi Profit_Category dari dataset
Global Skincare & Beauty E-Store, lalu menyimpan seluruh artifact yang
dibutuhkan oleh aplikasi Streamlit (app.py) ke folder ./artifacts.

Jalankan sekali saja (atau setiap kali ada data baru):
    python train_model.py

Artifact yang dihasilkan di folder ./artifacts:
    - model.pkl              -> Random Forest (tuned) sebagai model utama
    - scaler.pkl              -> StandardScaler yang sudah di-fit
    - label_encoder.pkl       -> LabelEncoder untuk target Profit_Category
    - feature_columns.pkl     -> urutan kolom fitur hasil one-hot encoding
    - all_models.pkl          -> dict berisi 4 model (LR, DT, RF, SVM) untuk halaman perbandingan
    - model_comparison.csv    -> tabel metrik perbandingan 4 model
    - ui_metadata.json        -> daftar pilihan dropdown & rentang nilai untuk form input di app
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_URL = os.environ.get(
    "DATA_URL",
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vQHKCQ5-LS4nL7Ger0OE2kxachW17ueOdGL0EwbDsNzctoxBWZdDCtnlCerAV7rFA/pub?output=csv",
)

ARTIFACT_DIR = "artifacts"
CATEGORICAL_COLS = ["Segment", "Region", "Market", "Subcategory", "Category"]
DROP_COLS = [
    "Row ID", "Order ID", "Customer ID", "Order Date",
    "City", "State", "Country", "Product",
    "Country latitude", "Country longitude",
    "Profit", "Profit_Category",
]


def load_and_clean_data(source: str) -> pd.DataFrame:
    df = pd.read_csv(source)

    df["Discount"] = pd.to_numeric(
        df["Discount"].astype(str).str.replace(",", "."), errors="coerce"
    )
    df["Profit"] = pd.to_numeric(
        df["Profit"].astype(str).str.replace(",", "."), errors="coerce"
    )

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def add_target(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    median_profit = df["Profit"].median()

    def label_profit(profit):
        if profit <= 0:
            return "Rugi"
        elif profit <= median_profit:
            return "Untung Rendah"
        else:
            return "Untung Tinggi"

    df["Profit_Category"] = df["Profit"].apply(label_profit)
    return df, median_profit


def build_features(df: pd.DataFrame):
    df = df.copy()
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Order_Year"] = df["Order Date"].dt.year
    df["Order_Month"] = df["Order Date"].dt.month
    df["Order_DayOfWeek"] = df["Order Date"].dt.dayofweek

    X = df.drop(columns=DROP_COLS)
    y = df["Profit_Category"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True)
    return X, y_encoded, le


def train_all_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(kernel="rbf", random_state=42, probability=True),
    }

    results = []
    fitted = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test, average="weighted")
        recall = recall_score(y_test, y_pred_test, average="weighted")
        f1 = f1_score(y_test, y_pred_test, average="weighted")

        fitted[name] = model
        results.append([name, train_acc, test_acc, precision, recall, f1])

    results_df = pd.DataFrame(
        results,
        columns=["Model", "Train Accuracy", "Test Accuracy", "Precision", "Recall", "F1-Score"],
    ).sort_values(by="Test Accuracy", ascending=False)

    return fitted, results_df


def tune_random_forest(X_train, y_train):
    # max_depth dibatasi (TIDAK menyertakan None) supaya pohon tidak tumbuh
    # tanpa batas. Pohon tanpa batas bisa mencapai akurasi train 100% tapi
    # menghasilkan file model raksasa (50-100+ MB) yang tidak praktis untuk
    # di-deploy ke Streamlit Cloud / disimpan di GitHub.
    param_grid = {
        "n_estimators": [100, 150],
        "max_depth": [10, 15, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def build_ui_metadata(df_raw: pd.DataFrame, median_profit: float) -> dict:
    category_subcategory_map = (
        df_raw.groupby("Category")["Subcategory"]
        .unique()
        .apply(lambda arr: sorted(arr.tolist()))
        .to_dict()
    )

    metadata = {
        "segments": sorted(df_raw["Segment"].unique().tolist()),
        "regions": sorted(df_raw["Region"].unique().tolist()),
        "markets": sorted(df_raw["Market"].unique().tolist()),
        "categories": sorted(df_raw["Category"].unique().tolist()),
        "category_subcategory_map": category_subcategory_map,
        "quantity_min": int(df_raw["Quantity"].min()),
        "quantity_max": int(df_raw["Quantity"].max()),
        "sales_min": float(df_raw["Sales"].min()),
        "sales_max": float(df_raw["Sales"].max()),
        "discount_min": float(df_raw["Discount"].min()),
        "discount_max": float(df_raw["Discount"].max()),
        "median_profit": float(median_profit),
        "total_rows": int(df_raw.shape[0]),
    }
    return metadata


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    print("1. Memuat & membersihkan data ...")
    df = load_and_clean_data(DATA_URL)
    df, median_profit = add_target(df)
    print(f"   -> {df.shape[0]} baris, median profit = {median_profit}")

    print("2. Menyiapkan fitur (encoding, drop kolom cardinality tinggi) ...")
    X, y, le = build_features(df)
    print(f"   -> Jumlah fitur akhir: {X.shape[1]}")

    print("3. Train-test split & scaling ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("4. Melatih 4 model dasar (untuk halaman perbandingan metrik) ...")
    _, results_df = train_all_models(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    print(results_df)

    print("5. Hyperparameter tuning Random Forest (model utama untuk prediksi) ...")
    best_rf, best_params, best_score = tune_random_forest(X_train_scaled, y_train)
    print(f"   -> Best params: {best_params}, CV score: {best_score:.4f}")

    test_acc_tuned = accuracy_score(y_test, best_rf.predict(X_test_scaled))
    print(f"   -> Test accuracy (tuned RF): {test_acc_tuned:.4f}")

    print("6. Menyimpan artifact ...")
    joblib.dump(best_rf, os.path.join(ARTIFACT_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    joblib.dump(le, os.path.join(ARTIFACT_DIR, "label_encoder.pkl"))
    joblib.dump(list(X.columns), os.path.join(ARTIFACT_DIR, "feature_columns.pkl"))
    results_df.to_csv(os.path.join(ARTIFACT_DIR, "model_comparison.csv"), index=False)

    ui_metadata = build_ui_metadata(df, median_profit)
    ui_metadata["best_rf_params"] = best_params
    ui_metadata["best_rf_cv_score"] = best_score
    ui_metadata["best_rf_test_accuracy"] = test_acc_tuned

    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_rf.feature_importances_,
    }).sort_values(by="Importance", ascending=False).head(15)
    feature_importance.to_csv(os.path.join(ARTIFACT_DIR, "feature_importance.csv"), index=False)

    with open(os.path.join(ARTIFACT_DIR, "ui_metadata.json"), "w") as f:
        json.dump(ui_metadata, f, indent=2)

    print(f"\nSelesai. Semua artifact tersimpan di folder '{ARTIFACT_DIR}/'.")


if __name__ == "__main__":
    main()

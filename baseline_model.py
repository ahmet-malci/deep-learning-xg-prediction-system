import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, precision_score, recall_score, \
    f1_score

warnings.filterwarnings('ignore')


def main():
    file_path = 'data/processed/shots_clean.csv'
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"HATA: {file_path} bulunamadı.")
        return

    X = df.drop(columns=['is_goal', 'match_id'], errors='ignore')
    y = df['is_goal']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss')
    }

    results = []

    print("\nGeleneksel Modeller Yarıştırılıyor (LR vs RF vs XGBoost)...\n")

    for name, model in models.items():
        print(f">>> {name} eğitiliyor...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "ROC-AUC": auc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })

    results_df = pd.DataFrame(results)
    display_df = results_df.copy()
    display_df['Accuracy'] = display_df['Accuracy'].apply(lambda x: f"%{x * 100:.2f}")
    display_df['ROC-AUC'] = display_df['ROC-AUC'].apply(lambda x: f"{x:.4f}")
    display_df['Precision'] = display_df['Precision'].apply(lambda x: f"%{x * 100:.2f}")
    display_df['Recall'] = display_df['Recall'].apply(lambda x: f"%{x * 100:.2f}")
    display_df['F1-Score'] = display_df['F1-Score'].apply(lambda x: f"{x:.4f}")

    print("\n" + "=" * 90)
    print("KARŞILAŞTIRMA TABLOSU")
    print("=" * 90)
    print(display_df.to_string(index=False))
    print("=" * 90)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x='Model',
        y='ROC-AUC',
        data=results_df,
        hue='Model',
        palette='viridis',
        legend=False
    )
    plt.ylim(0.5, 1.0)
    plt.title('Modellerin xG Tahmin Kalitesi', fontsize=14)
    plt.ylabel('ROC-AUC Skoru', fontsize=12)
    plt.xlabel('Model Türü', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(results_df['ROC-AUC']):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
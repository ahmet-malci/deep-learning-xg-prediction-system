import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


def main():
    file_path = Path("data/processed/shots_clean.csv")
    if not file_path.exists():
        print("HATA: shots_clean.csv bulunamadı.")
        return

    df = pd.read_csv(file_path)
    print("Özellikler güncelleniyor...")
    other_cols = [c for c in df.columns if 'play_pattern_Other' in c]
    df = df.drop(columns=other_cols, errors='ignore')

    fk_col = 'shot_type_Free Kick'
    op_col = 'shot_type_Open Play'


    if fk_col in df.columns and op_col in df.columns:
        df['active_shot_type'] = ((df[fk_col] == 1) | (df[op_col] == 1)).astype(int)
        df = df.drop(columns=[fk_col, op_col])
    print("Korelasyon matrisi çiziliyor...")
    corr_df = df.drop(columns=['match_id'], errors='ignore')

    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Güncellenmiş Veri Seti - Korelasyon Matrisi', fontsize=16)
    plt.tight_layout()
    plt.show()
    features = [c for c in df.columns if c not in ['match_id', 'is_goal']]
    X = df[features]
    y = df['is_goal']

    print(f"Modeller eğitiliyor... (Toplam Özellik: {len(features)})")

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='logloss')
    lr = LogisticRegression(random_state=42, max_iter=1000)

    rf.fit(X, y)
    xgb.fit(X, y)
    lr.fit(X, y)

    fi_df = pd.DataFrame({
        'Özellik': features,
        'Random Forest': rf.feature_importances_,
        'XGBoost': xgb.feature_importances_,
        'Logistic Regression': np.abs(lr.coef_[0])
    })

    scaler = MinMaxScaler()
    fi_normalized = fi_df.copy()
    fi_normalized[['Random Forest', 'XGBoost', 'Logistic Regression']] = scaler.fit_transform(
        fi_df[['Random Forest', 'XGBoost', 'Logistic Regression']]
    )

    fi_normalized['avg'] = fi_normalized.mean(axis=1, numeric_only=True)
    fi_melted = fi_normalized.melt(id_vars='Özellik', var_name='Model', value_name='Önem Derecesi')
    sorted_features = fi_normalized.sort_values('avg', ascending=False)['Özellik'].tolist()
    plt.figure(figsize=(15, 12))
    sns.barplot(x='Önem Derecesi', y='Özellik', hue='Model', data=fi_melted,
                palette='viridis', order=sorted_features)

    plt.title('Model Karşılaştırması', fontsize=16)
    plt.xlabel('Normalize Edilmiş Önem Derecesi (0-1)')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    print("\n--- NİHAİ ÖNEM SIRALAMASI (LK 10) ---")
    print(fi_normalized[['Özellik', 'avg']].sort_values('avg', ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
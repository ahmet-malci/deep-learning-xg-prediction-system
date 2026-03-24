import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def main():
    file_path = 'data/processed/shots_features.csv'
    try:
        df = pd.read_csv(file_path)
        print(f"Veri yüklendi. Toplam satır: {len(df)}")
    except FileNotFoundError:
        print(f"Hata: {file_path} bulunamadı!")
        return

    cols_to_drop = ['outcome', 'minute', 'second']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    if 'under_pressure' in df.columns:
        df['under_pressure'] = df['under_pressure'].fillna(0).astype(int)

    cat_cols = ['shot_type', 'body_part', 'technique', 'play_pattern']
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
            df[col] = le.fit_transform(df[col])

    if 'is_goal' not in df.columns:
        print("Hata: 'is_goal' hedef değişkeni veri setinde yok!")
        return

    X = df.drop(columns=['is_goal'])
    y = df['is_goal'].astype(int)
    X = X.select_dtypes(include=[np.number])

    X = X.fillna(0)

    plt.figure(figsize=(12, 8))
    corr_matrix = X.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Özellikler Arası Korelasyon Matrisi (Çoklu Doğrusallık Kontrolü)')
    plt.tight_layout()
    plt.show()

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    importances = rf_model.feature_importances_

    fi_df = pd.DataFrame({'Özellik': X.columns, 'Önem_Derecesi': importances})
    fi_df = fi_df.sort_values(by='Önem_Derecesi', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Önem_Derecesi', y='Özellik', data=fi_df, hue='Özellik', palette='viridis', legend=False)
    plt.title('Gole Etki Eden En Önemli Özellikler (Feature Importance)')
    plt.tight_layout()
    plt.show()

    print("\n--- ÖZELLİK ÖNEM SIRALAMASI ---")
    print(fi_df.to_string(index=False))


if __name__ == "__main__":
    main()
import pandas as pd
from pathlib import Path


def update_shots_csv():
    file_path = Path("data/processed/shots_clean.csv")

    if not file_path.exists():
        print("HATA: shots_clean.csv bulunamadı!")
        return

    df = pd.read_csv(file_path)
    print(f"İşlem öncesi sütun sayısı: {len(df.columns)}")

    other_cols = [c for c in df.columns if 'play_pattern_Other' in c]
    if other_cols:
        df = df.drop(columns=other_cols)
        print(f"{len(other_cols)} adet 'Other' sütunu temizlendi.")

    fk_col = 'shot_type_Free Kick'
    op_col = 'shot_type_Open Play'

    if fk_col in df.columns and op_col in df.columns:
        df['active_shot_type'] = ((df[fk_col] == 1) | (df[op_col] == 1)).astype(int)
        df = df.drop(columns=[fk_col, op_col])
        print("Free Kick ve Open Play -> 'active_shot_type' olarak birleştirildi.")

    df.to_csv(file_path, index=False)
    print(f"Veri seti güncellendi! Yeni sütun sayısı: {len(df.columns)}")


if __name__ == "__main__":
    update_shots_csv()
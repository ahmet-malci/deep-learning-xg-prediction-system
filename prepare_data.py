import pandas as pd
from pathlib import Path

def main():
    df = pd.read_csv('data/processed/shots_features.csv')

    df['keeper_pressure_ratio'] = df['keeper_distance'] / df['distance_to_goal']

    selected_columns = [
        'match_id',
        'distance_to_goal',
        'shot_angle_rad',
        'keeper_pressure_ratio',
        'nearest_defender_distance',
        'blocker_count',
        'defenders_count',
        'under_pressure',
        'body_part',
        'shot_type',
        'play_pattern',
        'is_goal'
    ]

    df_clean = df[selected_columns].copy()
    df_clean['keeper_pressure_ratio'] = df_clean['keeper_pressure_ratio'].fillna(-1)
    df_clean['nearest_defender_distance'] = df_clean['nearest_defender_distance'].fillna(-1)
    df_clean['under_pressure'] = df_clean['under_pressure'].fillna(0).astype(int)
    df_clean['blocker_count'] = df_clean['blocker_count'].fillna(0)

    cat_cols = ['body_part', 'shot_type', 'play_pattern']
    for col in cat_cols:
        df_clean[col] = df_clean[col].fillna('Unknown')

    df_clean = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)


    for col in df_clean.select_dtypes(include=['boolean', 'bool']).columns:
        df_clean[col] = df_clean[col].astype(int)

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / 'shots_clean.csv'

    df_clean.to_csv(output_filename, index=False)

    print(f"\nİşlem Başarılı! Nihai veri seti oluşturuldu: {output_filename}")
    print(f"Toplam sütun sayısı: {len(df_clean.columns)}")


if __name__ == "__main__":
    main()
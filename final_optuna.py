import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

print("🏆FİNAL OPTUNA (LİGHT GBM) (250 Trials)\n")

try:
    df = pd.read_csv("data/processed/shots_clean.csv")
except FileNotFoundError:
    print("❌ HATA: shots_clean.csv bulunamadı!")
    exit()

print("Fiziksel / Geometrik yeni sütunlar üretiliyor...")

df['angle_dist_ratio'] = df['shot_angle_rad'] / (df['distance_to_goal'] + 0.1)
df['angle_dist_mult'] = df['shot_angle_rad'] * df['distance_to_goal']

y = df['is_goal'].values
X = df.drop(columns=['is_goal'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

best_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.04679235350773043,
    'num_leaves': 71,
    'max_depth': 3,
    'scale_pos_weight': 1.6126671190985615,
    'feature_fraction': 0.9963737136891359,
    'verbose': -1
}

print("\nOptuna'nın bulduğu en iyi parametrelerle nihai model eğitiliyor...")
final_model = lgb.LGBMClassifier(**best_params, random_state=42, n_estimators=150)
final_model.fit(X_train, y_train)

test_probabilities = final_model.predict_proba(X_test)[:, 1]
best_threshold = 0.36

final_preds = (test_probabilities >= best_threshold).astype(int)

roc_auc = roc_auc_score(y_test, test_probabilities)
acc = accuracy_score(y_test, final_preds)
prec = precision_score(y_test, final_preds)
rec = recall_score(y_test, final_preds)
f1 = f1_score(y_test, final_preds)

tn, fp, fn, tp = confusion_matrix(y_test, final_preds).ravel()

print("\n" + "=" * 55)
print("FİNAL OPTUNA MODEL SONUÇLARI")
print("=" * 55)
print(f"Kullanılan Kusursuz Eşik : {best_threshold:.2f}")
print(f"Accuracy                 : %{acc * 100:.2f}")
print(f"ROC-AUC Skoru            : {roc_auc:.4f}")
print(f"Precision                : %{prec * 100:.2f}")
print(f"Recall                   : %{rec * 100:.2f}")
print(f"F1-Score                 : {f1:.4f} ")
print("=" * 55)

print("\n" + "=" * 55)
print("📊 KARMAŞIKLIK MATRİSİ (CONFUSION MATRIX)")
print("=" * 55)
print(f"✅ True Positive  (TP) [Model Gol Dedi, Gerçekte Gol Oldu]       : {tp}")
print(f"❌ False Positive (FP) [Model Gol Dedi, Gerçekte Gol Olmadı]     : {fp}")
print(f"✅ True Negative  (TN) [Model Gol Olmaz Dedi, Gerçekte Olmadı]   : {tn}")
print(f"❌ False Negative (FN) [Model Gol Olmaz Dedi, Gerçekte Gol Oldu] : {fn}")
print("=" * 55)
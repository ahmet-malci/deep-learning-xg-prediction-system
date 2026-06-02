import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

print("OPTUNA MODEL\n")

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

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 8.0),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'verbose': -1
    }

    gbm = lgb.LGBMClassifier(**params, random_state=42, n_estimators=150)
    gbm.fit(X_train, y_train)

    probs = gbm.predict_proba(X_test)[:, 1]

    best_f1_local = 0
    for thresh in np.arange(0.1, 0.9, 0.02):
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(y_test, preds, zero_division=0)
        if f1 > best_f1_local:
            best_f1_local = f1

    return best_f1_local

print("\n⚙Optuna Framework'ü Başlatılıyor... Bilgisayar en iyi hiperparametreleri arıyor!")
print("Lütfen bekleyin, birçok farklı model eğitilip test ediliyor...\n")


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print("=" * 60)
print(f"🎯 OPTUNA ARAMAYI BİTİRDİ! Bulunan En Yüksek F1: {study.best_value:.4f}")
print("En Kusursuz Parametreler:")
for key, value in study.best_params.items():
    print(f"  -> {key}: {value}")

print("\n--- Bulunan En İyi Ayarlarla Final Modeli Eğitiliyor ---")
final_model = lgb.LGBMClassifier(**study.best_params, random_state=42, n_estimators=150, verbose=-1)
final_model.fit(X_train, y_train)

test_probabilities = final_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, test_probabilities)

best_threshold = 0.5
best_f1 = 0
best_metrics = {}

for thresh in np.arange(0.01, 1.00, 0.01):
    y_pred_loop = (test_probabilities >= thresh).astype(int)
    if sum(y_pred_loop) == 0: continue

    current_f1 = f1_score(y_test, y_pred_loop)
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = thresh
        best_metrics = {
            'acc': accuracy_score(y_test, y_pred_loop),
            'prec': precision_score(y_test, y_pred_loop),
            'rec': recall_score(y_test, y_pred_loop)
        }

final_preds = (test_probabilities >= best_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, final_preds).ravel()

print("\n" + "=" * 55)
print("OPTUNA MODEL SONUÇLARI")
print("=" * 55)
print(f"Kullanılan Kusursuz Eşik : {best_threshold:.2f}")
print(f"Accuracy                 : %{best_metrics['acc'] * 100:.2f}")
print(f"ROC-AUC Skoru            : {roc_auc:.4f}")
print(f"Precision                : %{best_metrics['prec'] * 100:.2f}")
print(f"Recall                   : %{best_metrics['rec'] * 100:.2f}")
print(f"F1-Score                 : {best_f1:.4f}")
print("=" * 55)

print("\n" + "=" * 55)
print("📊 KARMAŞIKLIK MATRİSİ (CONFUSION MATRIX)")
print("=" * 55)
print(f"✅ True Positive  (TP) [Model Gol Dedi, Gerçekte Gol Oldu]       : {tp}")
print(f"❌ False Positive (FP) [Model Gol Dedi, Gerçekte Gol Olmadı]     : {fp}")
print(f"✅ True Negative  (TN) [Model Gol Olmaz Dedi, Gerçekte Olmadı]   : {tn}")
print(f"❌ False Negative (FN) [Model Gol Olmaz Dedi, Gerçekte Gol Oldu] : {fn}")
print("=" * 55)
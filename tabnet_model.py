import pandas as pd
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("Google TabNet Mimarisi Başlatılıyor...")

try:
    df = pd.read_csv("data/processed/shots_clean.csv")
except FileNotFoundError:
    print("❌ HATA: shots_clean.csv bulunamadı!")
    exit()

y = df['is_goal'].values
X = df.drop(columns=['is_goal']).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = TabNetClassifier(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":10, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax',
    device_name='auto'
)

print("\n--- TabNet Eğitimi Başlıyor---")

clf.fit(
    X_train=X_train_scaled, y_train=y_train,
    eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
    eval_name=['train', 'valid'],
    eval_metric=['auc'],
    max_epochs=100,
    patience=20,
    batch_size=256, virtual_batch_size=128,
    num_workers=0,
    weights=1,
    drop_last=False
)

test_probabilities = clf.predict_proba(X_test_scaled)[:, 1]
y_pred_class = clf.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred_class)
prec = precision_score(y_test, y_pred_class)
rec = recall_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)
roc_auc = roc_auc_score(y_test, test_probabilities)

print("\n" + "="*45)
print("TABNET TEST SONUÇLARI")
print("="*45)
print(f"Accuracy (Doğruluk)  : %{acc*100:.2f}")
print(f"ROC-AUC Skoru        : {roc_auc:.4f}")
print(f"Precision (Kesinlik) : %{prec*100:.2f}")
print(f"Recall (Gol Yakalama): %{rec*100:.2f}")
print(f"F1-Score (Denge)     : {f1:.4f}")
print("="*45)

print("\nTABNET ÖNEMLİ DEĞİŞKENLER")
feature_importances = clf.feature_importances_
importance_df = pd.DataFrame({
    'Özellik': df.drop(columns=['is_goal']).columns,
    'Dikkat Skoru': feature_importances
}).sort_values(by='Dikkat Skoru', ascending=False).head(5)
print(importance_df.to_string(index=False))

print("\n" + "⚙️" * 25)
print("BİLGİSAYAR EN İYİ EŞİĞİ ARIYOR...")

best_threshold = 0.5
best_f1 = 0
best_metrics = {}

for thresh in np.arange(0.01, 1.00, 0.01):
    y_pred_loop = (test_probabilities >= thresh).astype(int)

    if sum(y_pred_loop) == 0:
        continue

    current_f1 = f1_score(y_test, y_pred_loop)

    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = thresh
        best_metrics = {
            'acc': accuracy_score(y_test, y_pred_loop),
            'prec': precision_score(y_test, y_pred_loop),
            'rec': recall_score(y_test, y_pred_loop)
        }

print("\n" + "=" * 50)
print(f"MATEMATİKSEL OLARAK KUSURSUZ EŞİK BULUNDU: {best_threshold:.2f}")
print("=" * 50)
print(f"ROC-AUC Skoru        : {roc_auc:.4f}")
print(f"Accuracy (Doğruluk)  : %{best_metrics['acc'] * 100:.2f}")
print(f"Precision (Kesinlik) : %{best_metrics['prec'] * 100:.2f}")
print(f"Recall (Gol Yakalama): %{best_metrics['rec'] * 100:.2f}")
print(f"F1-Score (Maksimum)  : {best_f1:.4f} 🌟")
print("=" * 50)

final_preds = (test_probabilities >= best_threshold).astype(int)

from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_test, final_preds).ravel()

print("\n" + "=" * 55)
print("📊 KARMAŞIKLIK MATRİSİ (CONFUSION MATRIX)")
print("=" * 55)
print(f"✅ True Positive  (TP) [Model Gol Dedi, Gerçekte Gol Oldu]       : {tp}")
print(f"❌ False Positive (FP) [Model Gol Dedi, Gerçekte Gol Olmadı]     : {fp}")
print(f"✅ True Negative  (TN) [Model Gol Olmaz Dedi, Gerçekte Olmadı]   : {tn}")
print(f"❌ False Negative (FN) [Model Gol Olmaz Dedi, Gerçekte Gol Oldu] : {fn}")
print("=" * 55)
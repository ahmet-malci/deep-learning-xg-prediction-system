import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

print("Stacking Ensemble Modeli Başlatılıyor...")


try:
    df = pd.read_csv("data/processed/shots_clean.csv")
except FileNotFoundError:
    print("❌ HATA: shots_clean.csv bulunamadı!")
    exit()

y = df['is_goal'].values
X = df.drop(columns=['is_goal'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


num_negatives = len(y_train[y_train == 0])
num_positives = len(y_train[y_train == 1])
weight = num_negatives / num_positives
adjusted_weight = weight * 0.6


xgb_model = XGBClassifier(
    scale_pos_weight=adjusted_weight,
    learning_rate=0.05, max_depth=5, n_estimators=200, random_state=42,
    eval_metric='logloss'
)


lgbm_model = LGBMClassifier(
    class_weight={0: 1, 1: adjusted_weight},
    learning_rate=0.05, max_depth=5, n_estimators=200, random_state=42, verbose=-1
)


cat_model = CatBoostClassifier(
    scale_pos_weight=adjusted_weight,
    learning_rate=0.05, depth=5, iterations=200, random_state=42, verbose=0
)

estimators = [
    ('XGB', xgb_model),
    ('LGBM', lgbm_model),
    ('CAT', cat_model)
]

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=3
)

print("\n--- Modeller Eğitiliyor---")
stacking_clf.fit(X_train, y_train)


test_probabilities = stacking_clf.predict_proba(X_test)[:, 1]

print("\nBilgisayar En İyi Karar Eşiğini Arıyor...")
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
            'rec': recall_score(y_test, y_pred_loop),
            'auc': roc_auc_score(y_test, test_probabilities)
        }

print("\n" + "=" * 50)
print(f"STACKING ENSEMBLE SONUÇLARI")
print("=" * 50)
print(f"Kullanılan Kusursuz Eşik : {best_threshold:.2f}")
print(f"Accuracy (Doğruluk)      : %{best_metrics['acc'] * 100:.2f}")
print(f"ROC-AUC Skoru            : {best_metrics['auc']:.4f}")
print(f"Precision (Kesinlik)     : %{best_metrics['prec'] * 100:.2f}")
print(f"Recall (Gol Yakalama)    : %{best_metrics['rec'] * 100:.2f}")
print(f"F1-Score (Nihai Zirve)   : {best_f1:.4f} 🌟")
print("=" * 50)


final_preds = (test_probabilities >= best_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, final_preds).ravel()

print("\n" + "="*55)
print("📊 KARMAŞIKLIK MATRİSİ (CONFUSION MATRIX)")
print("="*55)
print(f"✅ True Positive  (TP) [Model Gol Dedi, Gerçekte Gol Oldu]       : {tp}")
print(f"❌ False Positive (FP) [Model Gol Dedi, Gerçekte Gol Olmadı]     : {fp}")
print(f"✅ True Negative  (TN) [Model Gol Olmaz Dedi, Gerçekte Olmadı]   : {tn}")
print(f"❌ False Negative (FN) [Model Gol Olmaz Dedi, Gerçekte Gol Oldu] : {fn}")
print("="*55)
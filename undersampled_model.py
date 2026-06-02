import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import resample


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


df = pd.read_csv("data/processed/shots_clean.csv")


df_goals = df[df.is_goal == 1]
df_no_goals = df[df.is_goal == 0]


df_no_goals_downsampled = resample(df_no_goals,
                                   replace=False,
                                   n_samples=len(df_goals),
                                   random_state=42)


df_balanced = pd.concat([df_goals, df_no_goals_downsampled])
print(f"Yeni Veri Seti Boyutu: {len(df_balanced)} (50/50 Dengelendi)")
# ------------------------------------

y = df_balanced['is_goal'].values
X = df_balanced.drop(columns=['is_goal']).values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_t = torch.FloatTensor(scaler.fit_transform(X_train)).to(device)
y_train_t = torch.FloatTensor(y_train).view(-1, 1).to(device)

original_X = df.drop(columns=['is_goal']).values
original_y = df['is_goal'].values
_, X_test_orig, _, y_test_orig = train_test_split(original_X, original_y, test_size=0.2, random_state=42, stratify=original_y)

X_test_t = torch.FloatTensor(scaler.transform(X_test_orig)).to(device)
y_test_t = torch.FloatTensor(y_test_orig).view(-1, 1).to(device)

class Undersampled_MLP(nn.Module):
    def __init__(self, input_dim):
        super(Undersampled_MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)

model = Undersampled_MLP(X.shape[1]).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n--- Budanmış Veri ile Eğitim Başlıyor ---")
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_train_t), y_train_t)
    loss.backward()
    optimizer.step()

# 6. Sonuçlar
model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(X_test_t)).cpu().numpy()
    preds = (probs >= 0.5).astype(int)


    acc = accuracy_score(y_test_orig, preds)
    roc_auc = roc_auc_score(y_test_orig, probs)
    prec = precision_score(y_test_orig, preds)
    rec = recall_score(y_test_orig, preds)
    f1 = f1_score(y_test_orig, preds)

    print("\n" + "=" * 45)
    print("UNDERSAMPLED (BUDANMIŞ) MLP SONUÇLARI")
    print("=" * 45)
    print(f"Accuracy (Doğruluk)  : %{acc * 100:.2f}")
    print(f"ROC-AUC Skoru        : {roc_auc:.4f}")
    print(f"Precision (Kesinlik) : %{prec * 100:.2f}")
    print(f"Recall (Gol Yakalama): %{rec * 100:.2f}")
    print(f"F1-Score (Denge)     : {f1:.4f}")
    print("=" * 45)

    tn, fp, fn, tp = confusion_matrix(y_test_orig, preds).ravel()

    print("\n" + "=" * 55)
    print("📊 KARMAŞIKLIK MATRİSİ (CONFUSION MATRIX)")
    print("=" * 55)
    print(f"✅ True Positive  (TP) [Model Gol Dedi, Gerçekte Gol Oldu]       : {tp}")
    print(f"❌ False Positive (FP) [Model Gol Dedi, Gerçekte Gol Olmadı]     : {fp}")
    print(f"✅ True Negative  (TN) [Model Gol Olmaz Dedi, Gerçekte Olmadı]   : {tn}")
    print(f"❌ False Negative (FN) [Model Gol Olmaz Dedi, Gerçekte Gol Oldu] : {fn}")
    print("=" * 55)
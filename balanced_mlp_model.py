import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

df = pd.read_csv("data/processed/shots_clean.csv")
y = df['is_goal'].values
X = df.drop(columns=['is_goal']).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Dengeleme Öncesi Gol Sayısı: {sum(y_train)} / Toplam: {len(y_train)}")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"Dengeleme Sonrası Gol Sayısı: {sum(y_train_balanced)} / Toplam: {len(y_train_balanced)}")
# ---------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train_scaled).to(device)
y_train_t = torch.FloatTensor(y_train_balanced).view(-1, 1).to(device)
X_test_t = torch.FloatTensor(X_test_scaled).to(device)
y_test_t = torch.FloatTensor(y_test).view(-1, 1).to(device)


class Balanced_xG_MLP(nn.Module):
    def __init__(self, input_dim):
        super(Balanced_xG_MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)

model = Balanced_xG_MLP(X_train.shape[1]).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

epochs = 300
print(f"\n--- Dengeli Veri ile {epochs} Epoch Eğitim Başlıyor ---")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_t)
    loss = criterion(predictions, y_train_t)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(X_test_t)).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    y_true = y_test_t.cpu().numpy()

    print("\n" + "="*45)
    print("DENGELİ (SMOTE) MLP SONUÇLARI")
    print("="*45)
    print(f"Accuracy  : %{accuracy_score(y_true, preds)*100:.2f}")
    print(f"ROC-AUC   : {roc_auc_score(y_true, probs):.4f}")
    print(f"Precision : %{precision_score(y_true, preds)*100:.2f}")
    print(f"Recall    : %{recall_score(y_true, preds)*100:.2f}")
    print(f"F1-Score  : {f1_score(y_true, preds):.4f}")
    print("="*45)


    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    print("\n" + "=" * 55)
    print("📊 KARMAŞIKLIK MATRİSİ (CONFUSION MATRIX)")
    print("=" * 55)
    print(f"✅ True Positive  (TP) [Model Gol Dedi, Gerçekte Gol Oldu]       : {tp}")
    print(f"❌ False Positive (FP) [Model Gol Dedi, Gerçekte Gol Olmadı]     : {fp}")
    print(f"✅ True Negative  (TN) [Model Gol Olmaz Dedi, Gerçekte Olmadı]   : {tn}")
    print(f"❌ False Negative (FN) [Model Gol Olmaz Dedi, Gerçekte Gol Oldu] : {fn}")
    print("=" * 55)
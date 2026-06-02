import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

try:
    df = pd.read_csv("data/processed/shots_clean.csv")
    print("Veri seti (shots_clean.csv) başarıyla yüklendi.")
except FileNotFoundError:
    print("❌ HATA: shots_clean.csv bulunamadı! Lütfen önce prepare_data.py ve update_data.py dosyalarını çalıştırın.")
    exit()

y = df['is_goal'].values
X = df.drop(columns=['is_goal']).values

print(f"Modele giren toplam özellik (sütun) sayısı: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train_scaled).to(device)
y_train_t = torch.FloatTensor(y_train).view(-1, 1).to(device)
X_test_t = torch.FloatTensor(X_test_scaled).to(device)
y_test_t = torch.FloatTensor(y_test).view(-1, 1).to(device)

class Deep_xG_MLP(nn.Module):
    def __init__(self, input_dim):
        super(Deep_xG_MLP, self).__init__()
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
            nn.Dropout(0.2),


            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),


            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


input_dimension = X_train.shape[1]
model = Deep_xG_MLP(input_dimension).to(device)

num_negatives = len(y_train[y_train == 0])
num_positives = len(y_train[y_train == 1])
weight_for_goal = num_negatives / num_positives

adjusted_weight = weight_for_goal * 0.5
print(f"Dengelenmiş Sınıf Ağırlığı: {adjusted_weight:.2f} kat ceza")

pos_weight_tensor = torch.FloatTensor([adjusted_weight]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

epochs = 500
print(f"\n--- {epochs} Epoch'luk Derin Eğitim Başlıyor ---")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    predictions = model(X_train_t)
    loss = criterion(predictions, y_train_t)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    test_predictions_logits = model(X_test_t)

    test_probabilities = torch.sigmoid(test_predictions_logits).cpu().numpy()

    y_pred_class = (test_probabilities >= 0.5).astype(int)
    y_test_numpy = y_test_t.cpu().numpy()

    acc = accuracy_score(y_test_numpy, y_pred_class)
    prec = precision_score(y_test_numpy, y_pred_class)
    rec = recall_score(y_test_numpy, y_pred_class)
    f1 = f1_score(y_test_numpy, y_pred_class)
    roc_auc = roc_auc_score(y_test_numpy, test_probabilities)

    print("\n" + "=" * 45)
    print("DEEP MLP (5 KATMANLI) TEST SONUÇLARI")
    print("=" * 45)
    print(f"Accuracy (Doğruluk)  : %{acc * 100:.2f}")
    print(f"ROC-AUC Skoru        : {roc_auc:.4f} ")
    print(f"Precision (Kesinlik) : %{prec * 100:.2f}")
    print(f"Recall (Gol Yakalama): %{rec * 100:.2f}")
    print(f"F1-Score (Denge)     : {f1:.4f}")
    print("=" * 45)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_class).ravel()

    print("\n" + "=" * 55)
    print("📊 KARMAŞIKLIK MATRİSİ (CONFUSION MATRIX)")
    print("=" * 55)
    print(f"✅ True Positive  (TP) [Model Gol Dedi, Gerçekte Gol Oldu]       : {tp}")
    print(f"❌ False Positive (FP) [Model Gol Dedi, Gerçekte Gol Olmadı]     : {fp}")
    print(f"✅ True Negative  (TN) [Model Gol Olmaz Dedi, Gerçekte Olmadı]   : {tn}")
    print(f"❌ False Negative (FN) [Model Gol Olmaz Dedi, Gerçekte Gol Oldu] : {fn}")
    print("=" * 55)
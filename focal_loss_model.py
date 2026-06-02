import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import random
import os

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Focal Loss MLP Modeli Başlatılıyor...\n")

try:
    df = pd.read_csv("data/processed/shots_clean.csv")
except FileNotFoundError:
    print("HATA: shots_clean.csv bulunamadı!")
    exit()

y = df['is_goal'].values
X = df.drop(columns=['is_goal']).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train_scaled).to(device)
y_train_t = torch.FloatTensor(y_train).view(-1, 1).to(device)
X_test_t = torch.FloatTensor(X_test_scaled).to(device)
y_test_t = torch.FloatTensor(y_test).view(-1, 1).to(device)

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.float32)
        pt = torch.exp(-bce_loss)
        alpha_tensor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_tensor * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

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

model = Deep_xG_MLP(X_train.shape[1]).to(device)

criterion = BinaryFocalLoss(alpha=0.85, gamma=2.0)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_t)
    loss = criterion(predictions, y_train_t)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    test_predictions_logits = model(X_test_t)
    test_probabilities = torch.sigmoid(test_predictions_logits).cpu().numpy().flatten()
    y_test_numpy = y_test_t.cpu().numpy().flatten()

best_threshold = 0.5
best_f1 = 0
best_metrics = {}

roc_auc = roc_auc_score(y_test_numpy, test_probabilities)

for thresh in np.arange(0.01, 1.00, 0.01):
    y_pred_loop = (test_probabilities >= thresh).astype(int)
    if sum(y_pred_loop) == 0: continue

    current_f1 = f1_score(y_test_numpy, y_pred_loop)
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = thresh
        best_metrics = {
            'acc': accuracy_score(y_test_numpy, y_pred_loop),
            'prec': precision_score(y_test_numpy, y_pred_loop),
            'rec': recall_score(y_test_numpy, y_pred_loop)
        }

final_preds = (test_probabilities >= best_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test_numpy, final_preds).ravel()

print("\n" + "=" * 45)
print("FOCAL LOSS MODEL RESULTS")
print("=" * 45)
print(f"Karar Eşiği : {best_threshold:.2f}")
print(f"Accuracy  : %{best_metrics['acc'] * 100:.2f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"Precision : %{best_metrics['prec'] * 100:.2f}")
print(f"Recall    : %{best_metrics['rec'] * 100:.2f}")
print(f"F1-Score  : {best_f1:.4f}")
print("=" * 45)


print("\n" + "=" * 45)
print("📊 KARMAŞIKLIK MATRİSİ (CONFUSION MATRIX)")
print("=" * 55)
print(f"✅ True Positive  (TP) [Model Gol Dedi, Gerçekte Gol Oldu]       : {tp}")
print(f"❌ False Positive (FP) [Model Gol Dedi, Gerçekte Gol Olmadı]     : {fp}")
print(f"✅ True Negative  (TN) [Model Gol Olmaz Dedi, Gerçekte Olmadı]   : {tn}")
print(f"❌ False Negative (FN) [Model Gol Olmaz Dedi, Gerçekte Gol Oldu] : {fn}")
print("=" * 45)
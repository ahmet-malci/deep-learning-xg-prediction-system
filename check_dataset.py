import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/processed/shots_clean.csv')
X = df.drop(columns=['is_goal', 'match_id'], errors='ignore')
y = df['is_goal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_rate = y_train.mean()
test_rate = y_test.mean()

print(f"Toplam Veri Seti Gol Oranı: %{y.mean()*100:.2f}")
print(f"Train Seti Gol Oranı:       %{train_rate*100:.2f}")
print(f"Test Seti Gol Oranı:        %{test_rate*100:.2f}")


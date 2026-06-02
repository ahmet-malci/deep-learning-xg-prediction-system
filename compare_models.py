import pandas as pd


def show_final_results():

    results = [
        {
            "Model Adı": "Deep MLP (Ana Model)",
            "F1-Score": 0.4190,
            "ROC-AUC": 0.8055,
            "Accuracy (%)": 84.78,
            "Precision (%)": 32.84,
            "Recall (%)": 57.89,
            "TP": 88,
            "FP": 180,
            "TN": 1271,
            "FN": 64
        },
        {
            "Model Adı": "Stacking Ensemble (0.28 Eşik)",
            "F1-Score": 0.3931,
            "ROC-AUC": 0.8142,
            "Accuracy (%)": 89.02,
            "Precision (%)": 41.30,
            "Recall (%)": 37.50,
            "TP": 57,
            "FP": 81,
            "TN": 1370,
            "FN": 95
        },
        {
            "Model Adı": "Balanced MLP (SMOTE / Yapay Veri)",
            "F1-Score": 0.3736,
            "ROC-AUC": 0.7933,
            "Accuracy (%)": 82.84,
            "Precision (%)": 28.57,
            "Recall (%)": 53.95,
            "TP": 82,
            "FP": 205,
            "TN": 1246,
            "FN": 70
        },
        {
            "Model Adı": "Undersampled MLP (Budanmış Veri)",
            "F1-Score": 0.3662,
            "ROC-AUC": 0.8174,
            "Accuracy (%)": 75.17,
            "Precision (%)": 24.16,
            "Recall (%)": 75.66,
            "TP": 112,
            "FP": 361,
            "TN": 1090,
            "FN": 37
        },
        {
            "Model Adı": "TabNet (0.62 Eşik)",
            "F1-Score": 0.3653,
            "ROC-AUC": 0.8005,
            "Accuracy (%)": 78.98,
            "Precision (%)": 25.59,
            "Recall (%)": 63.82,
            "TP": 97,
            "FP": 282,
            "TN": 1169,
            "FN": 55
        },
        {
            "Model Adı": "Focal Loss MLP (Gelişmiş Kayıp Fonk.)",
            "F1-Score": 0.4107,
            "ROC-AUC": 0.8088,
            "Accuracy (%)": 87.65,
            "Precision (%)": 37.50,
            "Recall (%)": 45.39,
            "TP": 69,
            "FP": 115,
            "TN": 1336,
            "FN": 83
        },
        {
            "Model Adı": "LightGBM (250 Trials)",
            "F1-Score": 0.4354,
            "ROC-AUC": 0.8162,
            "Accuracy (%)": 90.46,
            "Precision (%)": 49.58,
            "Recall (%)": 38.82,
            "TP": 59,
            "FP": 60,
            "TN": 1391,
            "FN": 93
        },

    ]


    df = pd.DataFrame(results)


    df = df.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)


    print("\n" + "=" * 125)
    print("MODEL KARŞILAŞTIRMA TABLOSU ".center(125))
    print("=" * 125)

    print(df.to_string(index=False))


    print("\n" + "=" * 125)
    print("📚 METRİKLER SÖZLÜĞÜ VE FORMÜLLER 📚".center(125))
    print("=" * 125)

    print("\n📌 KARMAŞIKLIK MATRİSİ (CONFUSION MATRIX) ELEMANLARI:")
    print("  ✅ TP (True Positive)  : Model 'GOL OLUR' dedi ve GERÇEKTE DE GOL OLDU.")
    print("  ❌ FP (False Positive) : Model 'GOL OLUR' dedi ama GERÇEKTE GOL OLMADI.")
    print("  ✅ TN (True Negative)  : Model 'GOL OLMAZ' dedi ve GERÇEKTE DE OLMADI.")
    print("  ❌ FN (False Negative) : Model 'GOL OLMAZ' dedi ama GERÇEKTE GOL OLDU.")

    print("\n📊 PERFORMANS METRİKLERİ:")
    print("  🎯 Precision (Kesinlik) : Formül -> TP / (TP + FP)")
    print("      Ne Anlatır? -> Modelin 'Gol olur' dediği şutların yüzde kaçı gerçekten gol oldu?")

    print("  🎣 Recall (Duyarlılık)  : Formül -> TP / (TP + FN)")
    print("      Ne Anlatır? -> Gerçekte atılmış olan tüm gollerin yüzde kaçını modelimiz önceden sezebildi?")

    print("  ⚖️ F1-Score (Denge)     : Formül -> 2 * (Precision * Recall) / (Precision + Recall)")
    print(
        "      Ne Anlatır? -> Precision ve Recall'un harmonik ortalamasıdır.")

    print("  ✅ Accuracy (Doğruluk)  : Formül -> (TP + TN) / (TP + FP + TN + FN)")
    print("      Ne Anlatır? -> Modelin yaptığı tüm olumlu ve olumsuz tahminlerin yüzde kaçı doğru çıktı?")

    print("  📈 ROC-AUC Skoru        : (Eğri Altında Kalan Alan)")
    print(
        "      Ne Anlatır? -> Modelin şutları gol olma ihtimaline göre sıraya dizme yeteneğidir. Rastgele bir gol şutuna, kaçan bir şuttan daha yüksek olasılık verme ihtimalidir.")
    print("=" * 125 + "\n")



if __name__ == "__main__":
    show_final_results()
# Deep Learning-based Expected Goals (xG) Prediction System

This project aims to develop a Deep Learning model to estimate the **Expected Goals (xG)** value of football shots using high-fidelity event data.

##  Project Overview
The model analyzes various features such as distance, angle, and shot type to predict the probability of a shot resulting in a goal.

## 📈 Current Progress
- [x] **Data Sourcing:** Obtained event data from [StatsBomb Open Data](https://github.com/statsbomb/open-data).
- [x] **Data Preprocessing:** Cleaned raw data and performed feature engineering.
- [x] **Baseline Model:** Developed an initial baseline model for performance benchmarking.
- [ ] **Neural Network:** Deep Learning architecture implementation in progress.

## 📂 File Structure
* `baseline_model.py`: Script for the initial baseline prediction model.
* `shots_clean.csv`: Preprocessed dataset ready for training.
* `shots_features.csv`: Engineered features including distance and angle calculations.

## 🛠 Tech Stack
* **Language:** Python 3.12 
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, XGBoost
* **Visualization:** Matplotlib, Seaborn
* **Data Source:** StatsBomb

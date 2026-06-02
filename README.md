# Deep Learning-based Expected Goals (xG) Prediction System

This project aims to develop a Deep Learning and Advanced Machine Learning pipeline to estimate the **Expected Goals (xG)** value of football shots using high-fidelity event data.

## 🚀 Project Overview
The model analyzes spatial (distance, angle) and contextual (shot type, pressure) features to predict the probability of a shot resulting in a goal. To overcome the severe class imbalance in football (goals are rare), this project implements custom Deep Learning architectures (Binary Focal Loss) and mathematically optimized tree-based models (Optuna + LightGBM).

## 📈 Current Progress
- [x] **Data Sourcing:** Obtained and parsed nested JSON event data from [StatsBomb Open Data](https://github.com/statsbomb/open-data).
- [x] **Data Preprocessing:** Cleaned raw data, removed noise (e.g., penalties), and engineered advanced geometric features.
- [x] **Baseline & Experimental Models:** Developed traditional baselines and tested experimental architectures (TabNet, Stacking Ensembles, Undersampling).
- [x] **Deep Learning (Neural Network):** Implemented a PyTorch MLP architecture with a custom `Binary Focal Loss` function to dynamically penalize easy misclassifications.
- [x] **Hyperparameter Optimization:** Utilized the Optuna framework (TPE algorithm) over 250 trials to mathematically find the global maximum F1-Score.
- [x] **Final Production Model:** Deployed a highly optimized LightGBM model utilizing a dynamically calculated decision threshold (Threshold = 0.36) to maximize precision.

## 📂 File Structure Architecture

### 1. Data Pipeline
* `download_data.py` / `update_data.py`: Scripts to fetch and update raw JSON event data from the StatsBomb API.
* `prepare_data.py` / `build_shots_dataset.py`: Cleans nested data, handles missing values, and compiles the core dataset.
* `check_dataset.py`: Exploratory Data Analysis (EDA) and dataset validation.
* `shots_clean.csv` / `shots_features.csv`: Processed datasets ready for model training.

### 2. Feature Engineering
* `feature_selection.py` / `feature_selection2.py`: Scripts calculating spatial geometries (Euclidean distance, angles) and engineering interaction features.

### 3. R&D and Experimental Models
* `baseline_model.py`: Initial logistic regression and traditional tree models.
* `undersampled_model.py`: Experimental model testing majority class undersampling algorithms.
* `ensemble_model.py`: Stacking ensemble architecture combining multiple tree algorithms.
* `tabnet_model.py`: Experimental deep tabular learning network.
* `mlp_model.py` / `balanced_mlp_model.py`: PyTorch-based Multilayer Perceptron testing environments.
* `compare_models.py`: Script to generate confusion matrices and compare evaluation metrics across different approaches.

### 4. The Champions (Final Models)
* `optuna_model.py`: The Bayesian Optimization lab running 250 trials to find the optimal LightGBM parameters.
* `focal_loss_model.py`: Deep Learning champion using PyTorch and a custom mathematical Binary Focal Loss (Alpha=0.85, Gamma=2.0).
* `final_optuna.py`: **Final Production Model** (LightGBM) featuring hardcoded optimum parameters and a custom 0.36 decision threshold.

## 🛠 Tech Stack
* **Language:** Python 3.12
* **Machine Learning:** LightGBM, XGBoost, Scikit-learn
* **Deep Learning:** PyTorch
* **Optimization Framework:** Optuna
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Data Source:** StatsBomb Open Data API

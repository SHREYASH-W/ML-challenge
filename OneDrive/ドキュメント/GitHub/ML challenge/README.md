# IEEE SB GEHU — ML Challenge: Device Fault Detection

## Project Description

This project was developed for the **IEEE SB GEHU ML Challenge** online qualifiers. The task is a binary classification problem aimed at detecting whether an embedded device is operating normally or experiencing a fault condition, based on 47 anonymized numerical features (F01–F47) captured by an internal monitoring system.

**Target Classes:**
- `0` — Device operating under normal conditions
- `1` — Device exhibiting a faulty condition

**Dataset Summary:**

| Split | Rows | Columns |
|-------|------|---------|
| TRAIN.csv | 43,776 | 48 (47 features + Class) |
| TEST.csv  | 10,944 | 48 (47 features + ID)    |

Class distribution in training data: **60.46% Normal / 39.54% Faulty**

**Model:** XGBoost (Extreme Gradient Boosting)  
XGBoost was chosen over deep learning because it consistently outperforms neural networks on structured/tabular data, trains faster, and requires less hyperparameter tuning.

**Validation Results:**
- Accuracy: **99%**
- ROC-AUC: **0.9993**

---

## Project Structure

```
ML-Challenge/
│
├── TRAIN.csv              # Training data (not included, provide your own)
├── TEST.csv               # Test data (not included, provide your own)
├── train_xgboost.py       # Main training and prediction script
├── FINAL.csv              # Output predictions (generated after running)
├── confusion_matrix.png   # Confusion matrix plot (generated after running)
├── feature_importance.png # Top 20 feature importances (generated after running)
└── README.md              # This file
```

---

## Setup Instructions

### Requirements

- Python 3.8+
- Google Colab (recommended) or local environment

### Install Dependencies

```bash
pip install xgboost scikit-learn pandas numpy matplotlib seaborn
```

If running on **Google Colab**, all packages except xgboost are pre-installed. Run:

```python
!pip install xgboost
```

---

## Usage Instructions

### 1. Mount Google Drive (Colab only)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Place Dataset Files

Put `TRAIN.csv` and `TEST.csv` in your Google Drive under:

```
MyDrive/ML Challenge Dataset/
```

### 3. Run the Script

Upload `train_xgboost.py` to Colab and run:

```python
exec(open('train_xgboost.py').read())
```

Or simply copy-paste the script contents into a Colab cell and run it.

### 4. Output

After running, the following files will be saved to your Google Drive:

- `FINAL.csv` — Submission file in the required format (`ID`, `CLASS`)
- `confusion_matrix.png` — Validation confusion matrix
- `feature_importance.png` — Top 20 most important features

### 5. Submission Format

`FINAL.csv` will look like:

```
ID,CLASS
1,1
2,0
3,0
4,1
...
```

This is in the exact format required for evaluation, with the same number of rows and order as `TEST.csv`.

---

## How It Works

1. Loads and splits training data (85% train / 15% validation, stratified)
2. Automatically computes `scale_pos_weight` to handle class imbalance
3. Trains XGBoost with early stopping on validation AUC to prevent overfitting
4. Evaluates on validation set and prints accuracy, AUC, and classification report
5. Predicts on the test set and saves `FINAL.csv`

---

*This project was created for educational purposes as part of the IEEE SB GEHU ML Challenge.*

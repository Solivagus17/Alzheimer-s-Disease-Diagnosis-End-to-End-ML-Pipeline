# Alzheimer's Disease Detection

A machine learning pipeline for multi-class classification of Alzheimer's disease progression stages using clinical, cognitive, and neuroimaging features.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Results](#results)
- [Visualisations](#visualisations)
- [Requirements](#requirements)
- [Usage](#usage)
- [Key Design Decisions](#key-design-decisions)

---

## Problem Statement

Alzheimer's disease progresses through distinct stages before a clinical diagnosis is made. Early and accurate identification of these stages can significantly improve patient outcomes. This project builds a classification model to distinguish between five diagnostic categories using non-invasive clinical and MRI-derived features.

---

## Dataset

| Property | Value |
|---|---|
| Total samples | 1,737 |
| Total features | 371 |
| Training samples | 1,390 |
| Test samples | 347 |
| Target column | `Diagnosis` |

The train/test split is pre-defined in the dataset via the `Test_data` column and is honoured throughout the pipeline.

### Target Classes

| Label | Full Name | Count |
|---|---|---|
| CN | Cognitively Normal | 417 |
| SMC | Significant Memory Concern | 106 |
| EMCI | Early Mild Cognitive Impairment | 310 |
| LMCI | Late Mild Cognitive Impairment | 562 |
| AD | Alzheimer's Disease | 342 |

### Feature Groups

- **Cognitive scores** — CDRSB, MMSE, ADAS11, ADAS13, RAVLT (immediate, learning, forgetting)
- **Demographics** — Age, Gender, Education, Ethnicity, Race, Marital status
- **Genetic risk** — ApoE4 allele count
- **PET imaging** — FDG-PET SUVR
- **MRI volumetrics** — Hippocampus, Entorhinal, Fusiform, MidTemp, Ventricles, WholeBrain, Intracranial volume
- **Cortical parcellation** — Volume, Surface Area, Cortical Thickness across ~150 brain regions (left & right hemispheres)

---

## Project Structure

```
Alzheimer Detection/
│
├── Alzheimer_DataSet.csv
├── Alzheimer_Analysis.ipynb
└── README.md
```

---

## Pipeline

### 1. Data Loading & Inspection
- Load CSV, inspect shape, dtypes, and class distribution
- Identify 17 fully-null columns and flag for removal

### 2. Exploratory Data Analysis
- Class distribution across all five diagnosis groups
- Boxplots of key clinical features stratified by diagnosis
- Correlation heatmap of core clinical and neuroimaging features
- Pairplot of cognitive and neuroimaging feature interactions

### 3. Preprocessing
- Honour the pre-defined train/test split via `Test_data` column
- Drop identifier columns (`RID`, `Test_data`) and all fully-null columns
- Label encode categorical features — fit on train, applied to test
- Median imputation — fit on train only
- StandardScaler normalisation — fit on train only

### 4. Feature Selection
- Random Forest trained on all features to rank by importance
- Cumulative importance curve used to find the elbow
- Top 40 features selected

### 5. Model Training & Cross-Validation
- 5-fold Stratified K-Fold CV across multiple classifiers
- Best model selected by mean CV accuracy

### 6. Hyperparameter Tuning
- `RandomizedSearchCV` on the best model
- Search run on top-40 selected features only

### 7. Test Set Evaluation
- Best tuned model evaluated once on the held-out test set
- Metrics: Accuracy, Precision, Recall, F1-score per class
- Confusion matrix — raw counts and row-normalised

### 8. Feature Importance
- Top 25 features from tuned model
- Prediction confidence distribution — correct vs incorrect predictions

---

## Results

| Property | Value |
|---|---|
| Best Model | Tuned Random Forest |
| Features Used | 40 (from 371 raw) |
| Cross-Validation | Stratified 5-Fold |
| Best CV Accuracy | 79.84% |
| Test Accuracy | 77.01% |
| Test Weighted F1 | 0.7516 |

### Top 5 Predictive Features

| Rank | Feature |
|---|---|
| 1 | CDRSB |
| 2 | MMSE |
| 3 | ADAS13 |
| 4 | ADAS11 |
| 5 | Volume (Cortical Parcellation) of LeftMedialOrbitofrontal |

---

## Visualisations

### Target Distribution
![Target Distribution]()

### EDA — Clinical Features by Diagnosis
![EDA Demographics]()

### Correlation Heatmap
![Correlation Heatmap]()

### Pairplot
![Pairplot]()

### Feature Importance (Pre-Tuning)
![Feature Importance]()

### Model Comparison
![Model Comparison]()

### Confusion Matrix
![Confusion Matrix]()

### Per-Class Metrics
![Per Class Metrics]()

### Prediction Confidence
![Prediction Confidence]()

### Feature Importance (Tuned Model)
![Tuned Feature Importance]()

---

## Requirements

```
python >= 3.12
numpy
pandas
matplotlib
seaborn
scikit-learn
```

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Usage

1. Clone the repository and place `Alzheimer_DataSet.csv` in the project root
2. Open `Alzheimer_Analysis.ipynb` in VS Code or Jupyter
3. Select Python 3.12 as the interpreter
4. Run all cells top to bottom

**Google Colab:**

```python
from google.colab import files
uploaded = files.upload()

df = pd.read_csv('/content/Alzheimer_DataSet.csv')
```

---

## Key Design Decisions

- **No data leakage** — imputer, scaler, and encoders are fit exclusively on the training set
- **Pre-defined split respected** — the `Test_data` flag in the dataset defines the official split
- **Data-driven feature selection** — top 40 features by Random Forest importance
- **Tuning on reduced features only** — keeps runtime under 5 minutes

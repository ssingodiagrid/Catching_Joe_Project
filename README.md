# Catching Joe — Browser Session User Identification

## Overview
This project solves the **“Catching Joe” problem**, where the goal is to identify whether a browser session belongs to a specific user **Joe (user_id = 0)** based on browsing behavior.

The dataset contains thousands of browsing sessions from multiple users. Because Joe represents only a very small fraction of the sessions, the task becomes a **highly imbalanced binary classification problem**.

The system learns Joe’s browsing patterns and predicts the **probability that a session belongs to Joe**.

---

# Problem Statement
Given browser session logs, determine whether a session belongs to **Joe**.

Each session contains information such as:

- browser type  
- operating system  
- locale  
- location  
- list of visited websites  
- time spent on each website  
- session timestamp  

Target variable:

1 → Joe session

0 → Other user session


---

# Dataset

| File | Description |
|-----|-------------|
| dataset.json | Training dataset with user identities |
| verify.json | Unlabeled sessions used for prediction |

Dataset characteristics:

- ~80,000 browser sessions  
- ~400 Joe sessions  
- Highly imbalanced dataset (~0.5% Joe sessions)

---

# Project Pipeline

## Project Workflow

```
Problem Understanding
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Baseline Model Training
        ↓
Model Improvement
        ↓
Final Joe Detection System
```


---

# Phase 1 — Problem Understanding
The task was formulated as a **binary classification problem** with the objective of detecting Joe’s sessions from browsing logs.

### Key challenges

- Extreme class imbalance  
- High-dimensional sparse feature space  
- Behavioral identification from browsing patterns  

---

# Phase 2 — Exploratory Data Analysis
EDA was performed to understand browsing behavior and identify useful signals.

### Key analyses

- session length distribution  
- most frequently visited websites  
- comparison between Joe and other users  
- temporal browsing patterns  
- device usage patterns  

Initial observations indicated that **website visitation patterns were the strongest behavioral indicators**.

---

# Phase 3 — Feature Engineering

Raw browser sessions were converted into machine learning features.

## Website Features
Website sequences were transformed using **TF-IDF vectorization**, converting browsing sessions into numerical vectors.

Example:
``` mail.google.com slack.com youtube.com ```


---

## Behavioral Session Features

Additional features extracted:

- number of sites visited  
- total browsing time  
- average time per site  

---

## Temporal Features

From session timestamps:

- hour of day  
- weekday  

---

## Device Features

Categorical encoding for:

- browser  
- operating system  
- locale  

---

## Final Feature Representation

The final feature matrix combines:

TF-IDF website features:

device features

session statistics

temporal features


---

# Phase 4 — Baseline Model

The baseline model uses **Logistic Regression**.

Model configuration:

```python
LogisticRegression(
    class_weight="balanced",
    max_iter=5000
)
```

## Why Logistic Regression?

- Performs well with **high-dimensional sparse data**
- Efficient for **TF-IDF features**
- Provides **interpretable coefficients**

### Evaluation Metrics

- Precision
- Recall
- F1-score
- ROC-AUC

---

# Phase 5 — Model Improvement

Several improvements were explored to enhance model performance.

### Threshold Optimization
Because of the **extreme class imbalance**, the classification threshold was tuned to better balance precision and recall.

### Hyperparameter Tuning
Grid search was applied to determine the optimal parameters for the Logistic Regression model.

### Model Comparison
An additional model (**LightGBM**) was trained for comparison.

### Results

- Logistic Regression achieved **much higher recall for Joe sessions**
- LightGBM **missed many Joe sessions**

Therefore, **Logistic Regression was selected as the final model**.

# Phase 6 — Final Joe Detection System

The final prediction pipeline:

```
Browser Session
      ↓
Feature Engineering
      ↓
TF-IDF Vectorization
      ↓
Device Encoding
      ↓
Session Statistics
      ↓
Logistic Regression
      ↓
Probability Prediction
      ↓
Threshold Decision
```

Output:```
Probability(session belongs to Joe)```

## Key Insights

### 1. Website patterns are the strongest signal
Certain websites frequently appeared in Joe’s sessions (e.g., `mail.google.com`, `slack.com`, `youtube.com`).  
These recurring patterns acted as behavioral fingerprints.

### 2. Device information is not unique
Joe used multiple browsers and operating systems, so device features were useful only as supporting signals.

### 3. Temporal features are weaker predictors
Joe’s sessions occurred at various times, making time-based features less significant.

### 4. Sparse TF-IDF features favor linear models
Because the dataset contains high-dimensional sparse features, **Logistic Regression performed better than tree-based models**.

---

## Model Insights

| Model | Observation |
|------|-------------|
| Logistic Regression | Very high recall for Joe sessions |
| LightGBM | Higher precision but missed many Joe sessions |

Since the objective is **reliable Joe detection**, **Logistic Regression was selected as the final model**.

---

## Example Prediction

**Example session**

```
mail.google.com → slack.com → youtube.com
```

**Model output**

```
Joe Probability = 0.87
Prediction = Joe
```

## Project Structure

```
catching-joe/
│
├── data/
│   ├── dataset.json
│   └── verify.json
│
├── notebooks/
│   └── joe_detection.ipynb
│
├── models/
│   ├── joe_detector_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── device_encoder.pkl
│
├── outputs/
│   └── joe_predictions.csv
│
└── README.md
```

---

## Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- TF-IDF Vectorization  
- Logistic Regression  
- Jupyter Notebook  

---

## Final Conclusion

This project demonstrates that **user browsing behavior can be used to identify individuals from browser sessions**.

Using **TF-IDF website features combined with Logistic Regression**, the system successfully learns Joe’s browsing patterns and predicts the probability that a session belongs to Joe.

This approach shows how **behavioral data can be leveraged for user identification and anomaly detection in large-scale browsing datasets**.
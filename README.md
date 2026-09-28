# Task 4 — Decision Tree Classifier (Bank Marketing Dataset)

Predicts whether a bank customer will subscribe to a term deposit (`y`: yes/no) based on demographic, campaign, and socio-economic features, using a Decision Tree classifier.

## Dataset

**Bank Marketing Dataset** (UCI Machine Learning Repository), created by S. Moro, P. Cortez, and P. Rita.

- Source: https://archive.ics.uci.edu/dataset/222/bank+marketing
- File used: `bank-additional/bank-additional/bank-additional-full.csv` (41,188 rows, 20 input features + target `y`)
- The file is semicolon-delimited (`;`) with quoted string fields.

Citation: S. Moro, P. Cortez and P. Rita. *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems, 2014.

## Repo Contents

- `bank_marketing_decision_tree.ipynb` — notebook with preprocessing pipeline, model training, and evaluation.
- `outputs/confusion_matrix.png` — confusion matrix on the test set.
- `outputs/feature_importance.png` — top 5 feature importances.
- `outputs/metrics.txt` — accuracy, precision, recall, and classification report.
- `app.py` — Streamlit app: interactive prediction form + model performance dashboard.
- `train_model.py` — trains the pipeline and saves it to `model/decision_tree_pipeline.joblib` for the app to load.
- `data/bank-additional-full.csv` — copy of the dataset, bundled so the app is self-contained for deployment.
- `requirements.txt` — dependencies for running the app / notebook.
- `README.md` — this file.

## Preprocessing

1. Loaded `bank-additional-full.csv` with `;` separator.
2. Dropped the `duration` column — per the dataset documentation, call duration is only known *after* the call takes place and leaks the outcome (`duration=0` implies `y="no"`), so it is unrealistic as a predictive feature and would inflate metrics artificially.
3. Categorical features (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`) were one-hot encoded; `"unknown"` values are kept as their own category rather than imputed, since they may carry signal (e.g., a customer refusing to disclose loan status).
4. Numeric features (`age`, `campaign`, `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`) were passed through unchanged (decision trees don't require scaling).
5. Target `y` was label-encoded (`yes` → 1, `no` → 0).
6. Data was split 80/20 into train/test sets, stratified on the target to preserve the class imbalance (~11% positive class) in both splits.

## Model

`sklearn.tree.DecisionTreeClassifier` (`max_depth=6`, `class_weight="balanced"`, `random_state=42`) inside a `ColumnTransformer` + `Pipeline`. `max_depth` was capped to keep the tree interpretable and reduce overfitting; `class_weight="balanced"` compensates for the ~89/11 class split so the minority ("subscribed") class isn't ignored.

## Results (test set, n=8,238)

| Metric | Value |
|---|---|
| Accuracy | 0.821 |
| Precision (yes) | 0.347 |
| Recall (yes) | 0.667 |

Confusion matrix:

| | Predicted no | Predicted yes |
|---|---|---|
| **Actual no** | 6143 | 1167 |
| **Actual yes** | 309 | 619 |

Top 5 features by importance:

| Rank | Feature | Importance |
|---|---|---|
| 1 | `nr.employed` (number of employees, national indicator) | 0.673 |
| 2 | `cons.conf.idx` (consumer confidence index) | 0.136 |
| 3 | `cons.price.idx` (consumer price index) | 0.043 |
| 4 | `euribor3m` (euribor 3-month rate) | 0.035 |
| 5 | `pdays` (days since last contact from a prior campaign) | 0.027 |

Full classification report in `outputs/metrics.txt`; plots in `outputs/confusion_matrix.png` and `outputs/feature_importance.png`. Re-run the notebook top-to-bottom to regenerate all outputs.

## Model Summary

A shallow (depth-6), class-balanced decision tree was trained on 19 demographic/campaign/economic features (call `duration` excluded as a leakage feature) to predict term-deposit subscription. The model reaches 82.1% accuracy, but because the target is heavily imbalanced (~89% no / ~11% yes), precision/recall on the minority ("yes") class are far more informative: recall of 0.667 means the model catches two-thirds of actual subscribers, at the cost of a lower precision of 0.347 (many false positives) — a direct result of using `class_weight="balanced"` to avoid the trivial "always predict no" model that would otherwise reach ~89% accuracy while catching zero subscribers. All top 5 features by importance are macro-economic/campaign-history indicators (`nr.employed` alone accounts for ~67% of total importance), consistent with the original Moro et al. (2014) finding that national economic context strongly affects subscription likelihood — customer demographics (age, job, education, etc.) contributed comparatively little to this tree's splits.

## Streamlit App

`app.py` provides an interactive UI: a **Predict** tab where you fill in a customer's details and get a subscribe/no-subscribe prediction with probability, and a **Model Performance** tab showing the metrics, confusion matrix, and feature-importance plot above.

### Run locally

```bash
pip install -r requirements.txt
python train_model.py       # trains the pipeline, saves model/decision_tree_pipeline.joblib
streamlit run app.py
```

### Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo (must include `app.py`, `requirements.txt`, `train_model.py`, `data/`, and `model/` if you want to skip retraining — or let Streamlit Cloud run `train_model.py` once via a startup step).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click **New app**.
3. Select this repo/branch and set **Main file path** to `app.py`.
4. Deploy. Streamlit Cloud installs `requirements.txt` automatically.

Note: `model/decision_tree_pipeline.joblib` is committed to the repo so the app loads instantly without retraining on every cold start; re-run `train_model.py` and commit the updated file if you change the preprocessing or model.

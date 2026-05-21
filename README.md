# Polimi@Oracle 2026 — Kaggle Competition

> **Organizer:** Lorenzo Barcella (Polimi Data Scientists x Oracle)
> **Type:** Binary Classification — Community Prediction Competition
> **Metric:** F1 Macro
> **Result:** top ~10% on the private leaderboard (private score: 0.7769)

---

## Objective

Predict whether a bank customer will purchase a financial product (`buyer = 1`) or not (`buyer = 0`), based on demographic, financial, and contact-history data.

The dataset contains **36,168 customers** in the training set and **9,043** in the test set, with 18 features covering:
- Demographic data (age, job, marital status, education)
- Financial data (balance, loans)
- Interaction history (contact channel, call duration, number of contacts)

The target class is **imbalanced**: only 11.7% of customers are buyers, which motivates the choice of F1 Macro as the evaluation metric.

---

## Results

| Approach | CV F1 Macro (OOF) | Public Score |
|---|---|---|
| LightGBM baseline | 0.7718 | — |
| Ensemble LGB + XGB + CatBoost | 0.7911 | 0.7769 |
| Ensemble v2 (Optuna + multi-seed + TE) | ~0.80+ | — |

---

## Project Structure

```
polimi-oracle/
├── README.md                         # this file
├── polimi_oracle_solution.py         # full solution code
├── EXPLANATION.md                    # detailed explanation of the code and the math
└── data/
    ├── train.csv                     # training set (not included — download from Kaggle)
    ├── test.csv                      # test set (not included — download from Kaggle)
    └── sample_submission.csv         # submission format
```

---

## Pipeline

```
Raw Data
   |
   v
Feature Engineering
   |-- Flag "never contacted before" (days_from_last_contact == -1)
   |-- Interaction features (duration x contacts, balance/age, ...)
   |-- Age binning
   |-- OOF Target Encoding (leak-free) for key categorical features
   |
   v
Training: 3 models in 5-fold CV x 3 seeds
   |-- LightGBM  (hyperparameters tuned with Optuna, 50 trials)
   |-- XGBoost
   |-- CatBoost  (native categorical feature handling)
   |
   v
Ensemble
   |-- Optimal weight search (grid search on OOF predictions)
   |-- Robust threshold optimization (median of per-fold thresholds)
   |
   v
submission.csv
```

---

## Technologies

- **Python 3.12**
- `lightgbm`, `xgboost`, `catboost` — gradient boosting models
- `optuna` — Bayesian hyperparameter tuning
- `scikit-learn` — CV, metrics, preprocessing
- `pandas`, `numpy`

---

## Data

The original data is owned by the competition and cannot be redistributed.
Download it directly from the Kaggle competition page:

[Polimi@Oracle — Kaggle Competition](https://www.kaggle.com/competitions/polimi-oracle)

Once downloaded, place the files in a `data/` folder:
```
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

---

## How to Reproduce

```bash
pip install lightgbm xgboost catboost optuna scikit-learn pandas numpy

# Download data from Kaggle (link above) and place in data/
# Paths are already set to data/ by default.
# If running on Kaggle, switch to the commented-out path in CELL 2 of the script.
python polimi_oracle_solution.py
```

The `submission.csv` file will be generated in the current directory.

---

## Further Improvements

The gap between this solution (0.7769) and the top of the leaderboard (~0.803) is roughly 2.5 points. The most likely strategies to close it, in order of expected impact for this specific dataset:

**1. Richer feature engineering**

The most important feature by a large margin was `last_contact_duration`. Stronger solutions likely built more sophisticated features around it, such as:
- Duration relative to the mean duration for the same `job` group
- Interactions between duration and `prev_camp_outcome`
- Aggregate statistics per group (e.g. mean purchase rate per `job` + `education` combination)

**2. Stacking**

Instead of a simple weighted average of the three models, a meta-model (e.g. logistic regression) is trained on the OOF predictions:

```
LGB OOF ──┐
XGB OOF ──┼──► Logistic Regression ──► final prediction
CAT OOF ──┘
```

This captures non-linear relationships between model outputs that a weighted average cannot express.

**3. More Optuna trials**

The tuning here ran for 50 trials. With 200–300 trials, hyperparameters improve meaningfully — especially for CatBoost, which carried the highest weight (0.80) in the final ensemble.

**4. Wider ensemble diversity**

Adding models such as `ExtraTreesClassifier` or sklearn's `HistGradientBoostingClassifier` increases ensemble diversity and tends to add useful signal beyond the three GBDT implementations used here.

**5. Tabular neural network**

Some top Kagglers include a TabNet or a simple MLP in the ensemble. Tree-based and neural models make sufficiently different errors that combining them reduces variance further.

---

## Documentation

For a detailed explanation of the code, algorithmic choices, and the math behind each component, see [`EXPLANATION.md`](EXPLANATION.md).

---

## Credits

- **Competition organizer:** [Lorenzo Barcella](https://www.kaggle.com/lorenzobarcella) — Polimi Data Scientists x Oracle
- **Competition page:** [Polimi@Oracle on Kaggle](https://www.kaggle.com/competitions/polimi-oracle)
- **Event:** Oracle Company Visit — Polimi Data Scientists, Milan, May 20, 2026

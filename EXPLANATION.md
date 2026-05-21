# Code Explanation — Polimi@Oracle

This document explains in detail every component of the pipeline, the math behind it, and the design choices made.

---

## Table of Contents

1. [The problem: imbalanced binary classification](#1-the-problem)
2. [Feature Engineering](#2-feature-engineering)
3. [OOF Target Encoding](#3-oof-target-encoding)
4. [Gradient Boosting: how it works](#4-gradient-boosting)
5. [The three models: LightGBM, XGBoost, CatBoost](#5-the-three-models)
6. [Optuna: Bayesian hyperparameter tuning](#6-optuna)
7. [Stratified Cross-Validation](#7-cross-validation)
8. [Multi-seed averaging](#8-multi-seed-averaging)
9. [Ensemble and weight optimization](#9-ensemble)
10. [Classification threshold optimization](#10-classification-threshold)
11. [Metric: F1 Macro](#11-f1-macro)

---

## 1. The Problem

The dataset contains bank customers with demographic and contact-history features. The **target** is `buyer` ∈ {0, 1}: did the customer purchase a financial product?

**The main challenge: class imbalance.**

```
buyer = 0  →  88.3% of customers
buyer = 1  →  11.7% of customers
```

A model that always predicts 0 would achieve 88% accuracy, but would be completely useless. This is why the chosen metric is **F1 Macro**, which evaluates performance on both classes equally.

---

## 2. Feature Engineering

Feature engineering transforms raw data into signals that are more useful for the model.

### 2.1 "Never contacted before" flag

```python
df['never_contacted'] = (df['days_from_last_contact'] == -1).astype(int)
df['days_clean']      = df['days_from_last_contact'].replace(-1, 999)
```

`days_from_last_contact = -1` is a sentinel value indicating no previous contact. Rather than leaving it as a number, we make it explicit with a binary flag and replace -1 with 999 (a large value meaning "a long time ago") in the numeric variable.

### 2.2 Interaction features

```python
df['contact_x_duration']   = df['number_of_contacts_performed'] * df['last_contact_duration']
df['prev_contacts_ratio']  = df['num_of_prev_contacts'] / (df['number_of_contacts_performed'] + 1)
df['balance_per_age']      = df['balance'] / (df['age'] + 1)
df['duration_per_contact'] = df['last_contact_duration'] / (df['number_of_contacts_performed'] + 1)
```

Tree-based models do not automatically create products or ratios between variables: if `duration × number_of_contacts` is a useful predictor, we must compute it explicitly. The `+ 1` in denominators prevents division by zero.

### 2.3 Age binning

```python
df['age_bin'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 100], labels=False)
```

Groups age into bands (under 30, 30–40, 40–50, 50–60, over 60). This helps the model capture non-linear patterns on age with less noise than the raw continuous value.

---

## 3. OOF Target Encoding

### The problem with naive target encoding

Suppose we want to encode the feature `job` as the mean probability of `buyer=1` for each occupation. If we compute this mean on the entire training set and then use it for training, the model sees the target labels in its input features — **data leakage**.

### The solution: Out-Of-Fold (OOF) encoding

```python
for tr_idx, val_idx in inner_kf.split(X, y):
    # Compute target mean ONLY on the training folds
    agg = tr_df.groupby(col)['y'].agg(['mean', 'count'])
    # Apply to the validation fold
    oof_te[val_idx] = X[col].iloc[val_idx].map(smooth).fillna(global_mean).values
```

For each fold, the target mean is computed on the remaining folds, never on the current one. This ensures the model never sees its own target during training.

### Bayesian smoothing

```python
smooth = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
```

If a category has very few examples (e.g. only 3 customers with `job = "student"`), its mean is unreliable. **Smoothing** balances the category mean with the global mean:

$$\hat{\mu}_c = \frac{n_c \cdot \mu_c + \lambda \cdot \mu_{global}}{n_c + \lambda}$$

where $n_c$ is the number of examples in the category, $\mu_c$ is the local mean, $\mu_{global}$ is the global mean, and $\lambda$ is the smoothing parameter (= 10 in the code).

- If $n_c \gg \lambda$: $\hat{\mu}_c \approx \mu_c$ (large category — trust the local mean)
- If $n_c \ll \lambda$: $\hat{\mu}_c \approx \mu_{global}$ (small category — shrink towards the global mean)

---

## 4. Gradient Boosting

All three models used (LightGBM, XGBoost, CatBoost) are implementations of **Gradient Boosted Decision Trees (GBDT)**.

### Core idea

The ensemble builds trees sequentially, where each new tree learns to correct the errors of the previous one.

Given the current model $F_m(x)$, the next tree $h_{m+1}$ is trained on the **negative gradient residuals**:

$$r_i = -\frac{\partial L(y_i, F_m(x_i))}{\partial F_m(x_i)}$$

The updated model is:

$$F_{m+1}(x) = F_m(x) + \eta \cdot h_{m+1}(x)$$

where $\eta$ is the **learning rate** (e.g. 0.05): it controls how much we correct at each step.

### Loss function for binary classification

For binary classification, the **log-loss** (binary cross-entropy) is used:

$$L(y, p) = -y \log(p) - (1 - y) \log(1 - p)$$

where $p \in [0,1]$ is the predicted probability. The model converts the raw tree output into a probability via the sigmoid function:

$$p = \sigma(F(x)) = \frac{1}{1 + e^{-F(x)}}$$

### `scale_pos_weight`

To handle class imbalance, a higher weight is assigned to the minority class:

```python
scale_pos_weight = (y == 0).sum() / (y == 1).sum()  # ≈ 7.5
```

This multiplies the loss of positive samples by 7.5, forcing the model to penalize errors on buyers more heavily.

---

## 5. The Three Models

### LightGBM

LightGBM introduces two key optimizations over classic XGBoost:

**Gradient-based One-Side Sampling (GOSS):** instead of using all samples to compute splits, it retains all samples with a large gradient (where the model is most wrong) and randomly samples those with a small gradient. This reduces computational cost while preserving accuracy.

**Exclusive Feature Bundling (EFB):** groups mutually exclusive features (that are rarely both non-zero) into a single feature, reducing the effective number of features.

It grows trees **leaf-wise** rather than level-wise: it always expands the leaf with the highest gain, producing deeper and more asymmetric trees that are often more accurate.

### XGBoost

XGBoost adds **explicit regularization** to the loss:

$$\mathcal{L} = \sum_i L(y_i, \hat{y}_i) + \sum_k \left[ \gamma T_k + \frac{1}{2}\lambda \|w_k\|^2 \right]$$

where $T_k$ is the number of leaves in tree $k$, $w_k$ are the leaf weights, $\gamma$ penalizes structural complexity, and $\lambda$ penalizes large weights (L2). It grows trees **level-wise**.

### CatBoost

CatBoost is specialized for categorical features. It uses a variant of target encoding called **Ordered Target Statistics** which — unlike classic target encoding — is intrinsically leak-free even without OOF, by using a random permutation of the data order.

In the code, we pass `cat_features=cat_indices` to activate this native mechanism.

---

## 6. Optuna

Optuna implements the **Tree-structured Parzen Estimator (TPE)**, a Bayesian optimization algorithm.

### Idea

Instead of exploring the parameter space randomly (random search), TPE builds a probabilistic model of which parameters yield good results:

1. Splits observed configurations into "good" ($l(x)$, top percentile) and "bad" ($g(x)$)
2. Approximates $l(x)$ and $g(x)$ with distributions (Parzen windows / kernel density estimation)
3. Samples the next point by maximizing the ratio $l(x) / g(x)$ (Expected Improvement)

In the code:
```python
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective_lgb, n_trials=50)
```

With 50 trials, Optuna efficiently explores the space of parameters such as `num_leaves`, `learning_rate`, `reg_alpha`, `reg_lambda`, and so on.

---

## 7. Cross-Validation

### Stratified K-Fold

With imbalanced data, standard cross-validation might by chance place all buyers in a single fold. **Stratified K-Fold** guarantees that each fold maintains the same class proportion as the original dataset.

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

With 5 folds: 4 folds for training (80% of the data) and 1 fold for validation (20%). The process repeats 5 times, once per validation fold.

### Out-Of-Fold (OOF) predictions

By accumulating predictions on the validation folds, we obtain an OOF prediction array of the same size as the training set:

```python
oof_lgb = np.zeros(len(X))
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    ...
    oof_lgb[val_idx] = model.predict_proba(X_val)[:, 1]
```

This is a reliable proxy for real test-set performance, without data leakage.

---

## 8. Multi-seed Averaging

```python
SEEDS = [42, 1337, 2024]
```

Decision trees have stochastic components (feature sampling, data sampling, initialization). With different seeds, the same model with the same parameters produces slightly different predictions.

Averaging predictions across multiple seeds is equivalent to implicit ensembling: it reduces variance without increasing bias. The effect is similar to training many more models at a contained computational cost.

$$\text{Final OOF} = \frac{1}{|\text{SEEDS}|} \sum_{s \in \text{SEEDS}} \text{OOF}_s$$

---

## 9. Ensemble

### Weighted average

The final predictions are a weighted average of the three models' probabilities:

$$p_{ensemble}(x) = w_1 \cdot p_{LGB}(x) + w_2 \cdot p_{XGB}(x) + w_3 \cdot p_{CAT}(x)$$

with $w_1 + w_2 + w_3 = 1$.

### Optimal weight search

Weights are found via grid search on the OOF predictions:

```python
for w1 in np.arange(0.10, 0.81, 0.05):
    for w2 in np.arange(0.10, 0.81, 0.05):
        w3 = 1 - w1 - w2
        blend = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
        score = f1_score(y, (blend >= thr).astype(int), average='macro')
```

### Why ensembling works

Each model captures different patterns in the data (different biases). By combining them, errors partially cancel out. Formally, if the models have correlation $\rho < 1$ with each other, the ensemble error is:

$$\text{Var}(\bar{p}) = \frac{\sigma^2}{n} \cdot [1 + (n-1)\rho]$$

The lower the correlation between models, the more the ensemble reduces variance.

---

## 10. Classification Threshold

The model outputs probabilities $p \in [0, 1]$. The final class is obtained by applying a threshold $\tau$:

$$\hat{y} = \begin{cases} 1 & \text{if } p \geq \tau \\ 0 & \text{otherwise} \end{cases}$$

The default threshold of 0.5 is often suboptimal with imbalanced classes. Lowering the threshold increases recall on the minority class at the cost of precision.

### Robust threshold (CV-based)

To avoid optimizing the threshold on the full OOF array (which introduces a small leakage), the code uses the **median of the per-fold optimal thresholds**:

```python
fold_thrs = []
for tr_idx, val_idx in skf_thr.split(blend_oof, y):
    # Find the optimal threshold on this training fold
    best_local = max([(f1_score(..., thr), thr) for thr in thresholds])
    fold_thrs.append(best_local[1])
robust_thr = np.median(fold_thrs)
```

The median is more robust than the mean to outliers: a fold with an unusual distribution does not shift the final threshold.

---

## 11. F1 Macro

### Precision and Recall

For the positive class (buyers):

$$\text{Precision} = \frac{TP}{TP + FP} \quad \text{(of all predicted buyers, how many actually are)}$$

$$\text{Recall} = \frac{TP}{TP + FN} \quad \text{(of all actual buyers, how many are found)}$$

### F1 Score

The harmonic mean of precision and recall:

$$F1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

The harmonic mean penalizes extreme values: a model with precision 1.0 and recall 0.0 will have F1 = 0, not 0.5.

### F1 Macro

Computes F1 separately for each class, then takes the unweighted average:

$$F1_{Macro} = \frac{F1_{\text{class 0}} + F1_{\text{class 1}}}{2}$$

Unlike F1 Micro (which weights by number of examples), F1 Macro treats both classes equally — essential with imbalanced data, to avoid ignoring the minority class.

---

## Summary of Design Choices

| Component | Choice | Rationale |
|---|---|---|
| Models | LGB + XGB + CatBoost | Diversity reduces ensemble variance |
| Imbalance handling | `scale_pos_weight` | Penalizes errors on the rare class more heavily |
| Categorical encoding | OOF Target Encoding + LabelEncoding | No leakage, captures ordinal signal |
| Tuning | Optuna TPE | More efficient than random search, no manual grid needed |
| Threshold | CV-based robust median | Avoids overfitting the threshold |
| Multi-seed | 3 seeds per model | Reduces variance, equivalent to 3x more models |
| CV | Stratified 5-fold | Preserves class proportions in every fold |

# =============================================================
# POLIMI @ ORACLE — Ensemble v2
# LightGBM (Optuna) + XGBoost + CatBoost
# Multi-seed averaging + OOF Target Encoding + Soglia robusta CV
# Metric: F1 Macro | Target atteso: 0.80–0.81
# =============================================================

# ── CELL 1: Import ────────────────────────────────────────────
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── CELL 2: Load Data ─────────────────────────────────────────
# Local (default): place data files in a data/ folder as described in README.md
# Kaggle:          replace with /kaggle/input/competitions/polimi-oracle/train.csv
train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')

print("Train shape:", train.shape)
print("Test shape: ", test.shape)
print("\nTarget distribution:")
print(train['buyer'].value_counts(normalize=True).round(3))

# ── CELL 3: Feature Engineering ───────────────────────────────
CAT_COLS = ['job', 'marital_status', 'education', 'was_in_default',
            'housing_loan', 'personal_loan', 'contact_channel',
            'month', 'prev_camp_outcome']

# Categoriche da target-encodare (medio-alta cardinalità o ordinabili)
TE_COLS = ['job', 'education', 'month', 'prev_camp_outcome']

def feature_engineering(df, train_ref):
    df = df.copy()

    # Flag e fix sentinel
    df['never_contacted'] = (df['days_from_last_contact'] == -1).astype(int)
    df['days_clean']      = df['days_from_last_contact'].replace(-1, 999)

    # Mese numerico
    month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
                 'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
    df['month_num'] = df['month'].map(month_map)

    # Feature di interazione
    df['contact_x_duration']  = df['number_of_contacts_performed'] * df['last_contact_duration']
    df['prev_contacts_ratio'] = df['num_of_prev_contacts'] / (df['number_of_contacts_performed'] + 1)
    df['balance_per_age']     = df['balance'] / (df['age'] + 1)
    df['duration_per_contact']= df['last_contact_duration'] / (df['number_of_contacts_performed'] + 1)
    df['is_long_call']        = (df['last_contact_duration'] > 300).astype(int)
    df['is_very_long_call']   = (df['last_contact_duration'] > 600).astype(int)
    df['has_prev_contacts']   = (df['num_of_prev_contacts'] > 0).astype(int)

    # Bin di età
    df['age_bin'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 100], labels=False)

    # Label encoding (concat per coprire categorie test-only)
    for col in CAT_COLS:
        le = LabelEncoder()
        le.fit(pd.concat([train_ref[col], df[col]]).astype(str))
        df[col] = le.transform(df[col].astype(str))

    df = df.drop(columns=['cust_id', 'days_from_last_contact'], errors='ignore')
    return df

X      = feature_engineering(train, train).drop(columns=['buyer'])
y      = train['buyer'].values
X_test = feature_engineering(test, train)
X_test = X_test[X.columns]  # allinea ordine colonne

print(f"\nFeatures dopo FE: {X.shape[1]}")

# ── CELL 4: OOF Target Encoding ──────────────────────────────
def oof_target_encoding(X, y, X_test, cols, n_splits=5, smoothing=10, seed=42):
    """
    OOF target encoding leak-free con smoothing bayesiano.
    - Train: per ogni fold si calcola la mean target SOLO sugli altri fold.
    - Test:  si usa la mean target su tutto il train.
    """
    X = X.copy()
    X_test = X_test.copy()
    global_mean = y.mean()
    inner_kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for col in cols:
        # OOF per il train
        oof_te = np.zeros(len(X))
        for tr_idx, val_idx in inner_kf.split(X, y):
            tr_df = pd.DataFrame({col: X[col].iloc[tr_idx].values, 'y': y[tr_idx]})
            agg   = tr_df.groupby(col)['y'].agg(['mean', 'count'])
            smooth = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
            oof_te[val_idx] = X[col].iloc[val_idx].map(smooth).fillna(global_mean).values
        X[f'{col}_te'] = oof_te

        # Full-train per il test
        full_df = pd.DataFrame({col: X[col].values, 'y': y})
        agg     = full_df.groupby(col)['y'].agg(['mean', 'count'])
        smooth  = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
        X_test[f'{col}_te'] = X_test[col].map(smooth).fillna(global_mean).values

    return X, X_test

X, X_test = oof_target_encoding(X, y, X_test, TE_COLS, n_splits=5, smoothing=10, seed=42)
print(f"Features dopo target encoding: {X.shape[1]}")

# ── CELL 5: Setup CV ─────────────────────────────────────────
N_SPLITS = 5
SEEDS    = [42, 1337, 2024]
spw      = (y == 0).sum() / (y == 1).sum()
print(f"\nScale pos weight: {spw:.2f}")

# Indici categoriche (per LGB e CatBoost native handling)
cat_indices = [X.columns.get_loc(c) for c in CAT_COLS]

# ── CELL 6: Optuna su LightGBM ───────────────────────────────
print("\n" + "="*60)
print("OPTUNA TUNING — LightGBM")
print("="*60)
N_TRIALS = 50

def objective_lgb(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'n_estimators': 2000,
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves':        trial.suggest_int('num_leaves', 15, 255),
        'max_depth':         trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'feature_fraction':  trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction':  trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq':      trial.suggest_int('bagging_freq', 1, 10),
        'reg_alpha':         trial.suggest_float('reg_alpha',  1e-8, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_split_gain':    trial.suggest_float('min_split_gain', 0.0, 1.0),
        'scale_pos_weight':  spw,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1,
    }

    skf_tune = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    for tr_idx, val_idx in skf_tune.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              categorical_feature=cat_indices,
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[val_idx] = m.predict_proba(X_val)[:, 1]

    # Soglia FISSA durante il tuning per evitare double-dipping
    return f1_score(y, (oof >= 0.4).astype(int), average='macro')

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
)
study.optimize(objective_lgb, n_trials=N_TRIALS, show_progress_bar=True)

best_lgb_params = study.best_params.copy()
best_lgb_params.update({
    'objective': 'binary',
    'n_estimators': 2000,
    'scale_pos_weight': spw,
    'verbose': -1,
    'n_jobs': -1,
})
print(f"\nBest LGB F1 (soglia=0.4): {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# ── CELL 7: Multi-seed training (LGB + XGB + CAT) ────────────
print("\n" + "="*60)
print("MULTI-SEED TRAINING")
print("="*60)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
tst_lgb = np.zeros(len(X_test))
tst_xgb = np.zeros(len(X_test))
tst_cat = np.zeros(len(X_test))

def run_lgb(seed):
    global oof_lgb, tst_lgb
    skf_s = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    params = best_lgb_params.copy()
    params['random_state'] = seed
    scores = []
    for fold, (tr_idx, val_idx) in enumerate(skf_s.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              categorical_feature=cat_indices,
              callbacks=[lgb.early_stopping(50, verbose=False)])
        p = m.predict_proba(X_val)[:, 1]
        oof_lgb[val_idx] += p / len(SEEDS)
        tst_lgb          += m.predict_proba(X_test)[:, 1] / (N_SPLITS * len(SEEDS))
        scores.append(f1_score(y_val, (p >= 0.4).astype(int), average='macro'))
    print(f"    LGB seed={seed}: F1 medio fold = {np.mean(scores):.4f}")

def run_xgb(seed):
    global oof_xgb, tst_xgb
    skf_s = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    scores = []
    for fold, (tr_idx, val_idx) in enumerate(skf_s.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        m = xgb.XGBClassifier(
            objective='binary:logistic', n_estimators=2000, learning_rate=0.03,
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0, gamma=0.0,
            scale_pos_weight=spw, random_state=seed,
            verbosity=0, eval_metric='logloss',
            early_stopping_rounds=50, n_jobs=-1,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        p = m.predict_proba(X_val)[:, 1]
        oof_xgb[val_idx] += p / len(SEEDS)
        tst_xgb          += m.predict_proba(X_test)[:, 1] / (N_SPLITS * len(SEEDS))
        scores.append(f1_score(y_val, (p >= 0.4).astype(int), average='macro'))
    print(f"    XGB seed={seed}: F1 medio fold = {np.mean(scores):.4f}")

def run_cat(seed):
    global oof_cat, tst_cat
    skf_s = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    scores = []
    for fold, (tr_idx, val_idx) in enumerate(skf_s.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        m = CatBoostClassifier(
            iterations=2000, learning_rate=0.03, depth=6,
            l2_leaf_reg=3.0, bagging_temperature=0.5,
            scale_pos_weight=spw, random_seed=seed,
            cat_features=cat_indices,
            verbose=0, early_stopping_rounds=50,
        )
        m.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        p = m.predict_proba(X_val)[:, 1]
        oof_cat[val_idx] += p / len(SEEDS)
        tst_cat          += m.predict_proba(X_test)[:, 1] / (N_SPLITS * len(SEEDS))
        scores.append(f1_score(y_val, (p >= 0.4).astype(int), average='macro'))
    print(f"    CAT seed={seed}: F1 medio fold = {np.mean(scores):.4f}")

for s in SEEDS:
    print(f"\n--- Seed {s} ---")
    run_lgb(s)
    run_xgb(s)
    run_cat(s)

# F1 individuali a soglia 0.4 di riferimento
print("\n=== F1 OOF individuali (soglia 0.4) ===")
for name, oof in [('LGB', oof_lgb), ('XGB', oof_xgb), ('CAT', oof_cat)]:
    s = f1_score(y, (oof >= 0.4).astype(int), average='macro')
    print(f"  {name}: {s:.4f}")

# ── CELL 8: Ricerca pesi e soglia ────────────────────────────
print("\n" + "="*60)
print("OTTIMIZZAZIONE PESI E SOGLIA")
print("="*60)

# Pesi ottimali (step largo per evitare overfit sul rumore OOF)
best_score, best_w, best_thr = 0, (1/3, 1/3, 1/3), 0.5
for w1 in np.arange(0.10, 0.81, 0.05):
    for w2 in np.arange(0.10, 0.81, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.10:
            continue
        blend = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
        for thr in np.arange(0.20, 0.65, 0.02):
            score = f1_score(y, (blend >= thr).astype(int), average='macro')
            if score > best_score:
                best_score, best_w, best_thr = score, (w1, w2, w3), thr

w1, w2, w3 = best_w
print(f"Pesi greedy:  LGB={w1:.2f} | XGB={w2:.2f} | CAT={w3:.2f}")
print(f"Soglia greedy: {best_thr:.3f}")
print(f"F1 ensemble (greedy): {best_score:.4f}")

# Soglia robusta CV-based: mediana delle soglie ottimali per fold
print("\n--- Soglia robusta (CV-based) ---")
blend_oof = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
skf_thr   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_thrs = []
for tr_idx, val_idx in skf_thr.split(blend_oof, y):
    best_local = (0, 0.4)
    for thr in np.arange(0.20, 0.65, 0.01):
        s = f1_score(y[tr_idx], (blend_oof[tr_idx] >= thr).astype(int), average='macro')
        if s > best_local[0]:
            best_local = (s, thr)
    fold_thrs.append(best_local[1])
robust_thr   = float(np.median(fold_thrs))
robust_score = f1_score(y, (blend_oof >= robust_thr).astype(int), average='macro')
print(f"Soglie per fold: {[round(t,3) for t in fold_thrs]}")
print(f"Soglia robusta (mediana): {robust_thr:.3f}")
print(f"F1 con soglia robusta: {robust_score:.4f}")

# Sceglie la soglia: preferisci robusta a meno che greedy sia molto meglio
if best_score - robust_score > 0.003:
    final_thr = best_thr
    print(f"→ Uso soglia greedy: {final_thr:.3f} (Δ favorevole = {best_score-robust_score:.4f})")
else:
    final_thr = robust_thr
    print(f"→ Uso soglia robusta: {final_thr:.3f}")

# ── CELL 9: Generazione Submission ───────────────────────────
test_blend  = w1 * tst_lgb + w2 * tst_xgb + w3 * tst_cat
final_preds = (test_blend >= final_thr).astype(int)

submission = pd.DataFrame({
    'cust_id': test['cust_id'],
    'buyer':   final_preds
})
submission.to_csv('submission.csv', index=False)

print("\n" + "="*60)
print("SUBMISSION GENERATA")
print("="*60)
print(f"File: submission.csv")
print(f"Buyer predetti: {final_preds.sum()} / {len(final_preds)} ({final_preds.mean():.1%})")
print(f"Distribuzione reale train: {y.mean():.1%}")
print(submission.head(10))

# Spiegazione del Codice — Polimi@Oracle

Questo documento spiega in dettaglio ogni componente del pipeline, la matematica che ci sta dietro e le scelte progettuali.

---

## Indice

1. [Il problema: classificazione binaria sbilanciata](#1-il-problema)
2. [Feature Engineering](#2-feature-engineering)
3. [OOF Target Encoding](#3-oof-target-encoding)
4. [Gradient Boosting: come funziona](#4-gradient-boosting)
5. [I tre modelli: LightGBM, XGBoost, CatBoost](#5-i-tre-modelli)
6. [Optuna: hyperparameter tuning bayesiano](#6-optuna)
7. [Cross-Validation stratificata](#7-cross-validation)
8. [Multi-seed averaging](#8-multi-seed-averaging)
9. [Ensemble e ottimizzazione dei pesi](#9-ensemble)
10. [Ottimizzazione della soglia di classificazione](#10-soglia-di-classificazione)
11. [Metrica: F1 Macro](#11-metrica-f1-macro)

---

## 1. Il problema

Il dataset contiene clienti bancari con feature demografiche e di storico contatti. Il **target** è `buyer` ∈ {0, 1}: il cliente ha acquistato un prodotto finanziario?

**Il problema principale: classe sbilanciata.**

```
buyer = 0  →  88.3% dei clienti
buyer = 1  →  11.7% dei clienti
```

Un modello che predice sempre 0 avrebbe accuracy del 88%, ma sarebbe completamente inutile. Per questo la metrica scelta è **F1 Macro**, che valuta le performance su entrambe le classi in modo equo.

---

## 2. Feature Engineering

Il feature engineering trasforma i dati grezzi in segnali più utili per il modello.

### 2.1 Flag "mai contattato prima"

```python
df['never_contacted'] = (df['days_from_last_contact'] == -1).astype(int)
df['days_clean']      = df['days_from_last_contact'].replace(-1, 999)
```

`days_from_last_contact = -1` è un valore sentinella che indica nessun contatto precedente. Invece di lasciarlo come numero, lo esplicitiamo con un flag binario e sostituiamo -1 con 999 (valore grande = "molto tempo fa") nella variabile numerica.

### 2.2 Feature di interazione

```python
df['contact_x_duration']   = df['number_of_contacts_performed'] * df['last_contact_duration']
df['prev_contacts_ratio']  = df['num_of_prev_contacts'] / (df['number_of_contacts_performed'] + 1)
df['balance_per_age']      = df['balance'] / (df['age'] + 1)
df['duration_per_contact'] = df['last_contact_duration'] / (df['number_of_contacts_performed'] + 1)
```

I modelli ad albero non creano automaticamente prodotti o rapporti tra variabili: se `durata × numero_contatti` è un buon predittore, dobbiamo calcolarlo esplicitamente. Il `+ 1` al denominatore evita divisioni per zero.

### 2.3 Binning dell'età

```python
df['age_bin'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 100], labels=False)
```

Raggruppa l'età in fasce (under 30, 30-40, 40-50, 50-60, over 60). Questo permette al modello di catturare pattern non lineari sull'età con meno rumore rispetto al valore continuo.

---

## 3. OOF Target Encoding

### Il problema con il target encoding naïve

Supponiamo di voler codificare la feature `job` come la probabilità media di `buyer=1` per ogni professione. Se calcoliamo questa media sull'intero training set e poi la usiamo per addestrare, il modello vede i label del target nei dati di input — **data leakage**.

### La soluzione: Out-Of-Fold (OOF) encoding

```python
for tr_idx, val_idx in inner_kf.split(X, y):
    # Calcola la media target SOLO sui fold di training
    agg = tr_df.groupby(col)['y'].agg(['mean', 'count'])
    # Applica al fold di validazione
    oof_te[val_idx] = X[col].iloc[val_idx].map(smooth).fillna(global_mean).values
```

Per ogni fold, la media target viene calcolata sui fold rimanenti, non sul fold corrente. Così il modello non vede mai il proprio target durante il training.

### Smoothing bayesiano

```python
smooth = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
```

Se una categoria ha pochi esempi (es. solo 3 clienti con `job = "student"`), la sua media è poco affidabile. Il **smoothing** bilancia la media della categoria con la media globale:

$$\hat{\mu}_c = \frac{n_c \cdot \mu_c + \lambda \cdot \mu_{global}}{n_c + \lambda}$$

dove $n_c$ è il numero di esempi nella categoria, $\mu_c$ è la media locale, $\mu_{global}$ è la media globale, e $\lambda$ è il parametro di smoothing (= 10 nel codice).

- Se $n_c \gg \lambda$: $\hat{\mu}_c \approx \mu_c$ (categoria grande, ci fidiamo della media locale)
- Se $n_c \ll \lambda$: $\hat{\mu}_c \approx \mu_{global}$ (categoria piccola, regressione verso la media)

---

## 4. Gradient Boosting

Tutti e tre i modelli usati (LightGBM, XGBoost, CatBoost) sono implementazioni di **Gradient Boosted Decision Trees (GBDT)**.

### Idea di base

L'ensemble costruisce alberi in sequenza, dove ogni nuovo albero impara a correggere gli errori del precedente.

Dato un modello corrente $F_m(x)$, il prossimo albero $h_{m+1}$ viene addestrato sui **residui negativi del gradiente**:

$$r_i = -\frac{\partial L(y_i, F_m(x_i))}{\partial F_m(x_i)}$$

Il modello aggiornato è:

$$F_{m+1}(x) = F_m(x) + \eta \cdot h_{m+1}(x)$$

dove $\eta$ è il **learning rate** (es. 0.05): controlla quanto "correggiamo" ad ogni step.

### Loss function per classificazione binaria

Per classificazione binaria si usa la **log-loss** (binary cross-entropy):

$$L(y, p) = -y \log(p) - (1 - y) \log(1 - p)$$

dove $p \in [0,1]$ è la probabilità predetta. Il modello converte l'output degli alberi in probabilità tramite la funzione sigmoide:

$$p = \sigma(F(x)) = \frac{1}{1 + e^{-F(x)}}$$

### `scale_pos_weight`

Per bilanciare le classi, si assegna un peso maggiore alla classe minoritaria:

```python
scale_pos_weight = (y == 0).sum() / (y == 1).sum()  # ≈ 7.5
```

Questo moltiplica la loss dei campioni positivi per 7.5, forzando il modello a penalizzare di più gli errori sui buyers.

---

## 5. I tre modelli

### LightGBM

LightGBM introduce due ottimizzazioni chiave rispetto a XGBoost classico:

**Gradient-based One-Side Sampling (GOSS):** invece di usare tutti i campioni per calcolare i split, mantiene tutti i campioni con gradiente grande (dove il modello sbaglia di più) e campiona casualmente quelli con gradiente piccolo. Questo riduce il costo computazionale mantenendo l'accuratezza.

**Exclusive Feature Bundling (EFB):** raggruppa feature mutuamente esclusive (che raramente sono entrambe non-zero) in un'unica feature, riducendo il numero di feature effettive.

Cresce alberi **leaf-wise** (per foglia) invece che level-wise: espande sempre la foglia con il massimo guadagno, producendo alberi più profondi e asimmetrici, spesso più accurati.

### XGBoost

XGBoost aggiunge alla loss una **regularizzazione esplicita**:

$$\mathcal{L} = \sum_i L(y_i, \hat{y}_i) + \sum_k \left[ \gamma T_k + \frac{1}{2}\lambda \|w_k\|^2 \right]$$

dove $T_k$ è il numero di foglie dell'albero $k$, $w_k$ sono i pesi delle foglie, $\gamma$ penalizza la complessità strutturale e $\lambda$ penalizza i pesi grandi (L2). Cresce alberi **level-wise**.

### CatBoost

CatBoost è specializzato nel gestire feature categoriche. Usa una variante del target encoding chiamata **Ordered Target Statistics** che — a differenza del target encoding classico — è intrinsecamente leak-free anche senza OOF, usando una permutazione casuale dell'ordine dei dati.

Nel codice passiamo `cat_features=cat_indices` per attivare questo meccanismo nativo.

---

## 6. Optuna

Optuna implementa la **Tree-structured Parzen Estimator (TPE)**, un algoritmo di ottimizzazione bayesiana.

### Idea

Invece di esplorare lo spazio dei parametri casualmente (random search), TPE costruisce un modello probabilistico di quali parametri danno buoni risultati:

1. Divide le configurazioni osservate in "buone" ($l(x)$, percentile superiore) e "cattive" ($g(x)$)
2. Approssima $l(x)$ e $g(x)$ con distribuzioni (Parzen windows / kernel density estimation)
3. Campiona il prossimo punto massimizzando il rapporto $l(x) / g(x)$ (Expected Improvement)

Nel codice:
```python
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective_lgb, n_trials=50)
```

Con 50 trial, Optuna esplora efficientemente lo spazio di parametri come `num_leaves`, `learning_rate`, `reg_alpha`, `reg_lambda`, ecc.

---

## 7. Cross-Validation

### Stratified K-Fold

Con dati sbilanciati, la cross-validation standard può per caso mettere tutti i buyers in un solo fold. La **Stratified K-Fold** garantisce che ogni fold abbia la stessa proporzione di classi del dataset originale.

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Con 5 fold: 4 fold per training (80% dei dati) e 1 fold per validation (20%). Il processo si ripete 5 volte, una per ogni fold di validation.

### Out-Of-Fold (OOF) predictions

Accumulando le predizioni sui fold di validation si ottiene un array di predizioni **OOF** della stessa dimensione del training set:

```python
oof_lgb = np.zeros(len(X))
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    ...
    oof_lgb[val_idx] = model.predict_proba(X_val)[:, 1]
```

Questo è un proxy affidabile della performance reale sul test set, senza data leakage.

---

## 8. Multi-seed averaging

```python
SEEDS = [42, 1337, 2024]
```

Gli alberi di decisione hanno componenti stocastiche (campionamento delle feature, dei dati, inizializzazione). Con seed diversi, lo stesso modello con gli stessi parametri produce predizioni leggermente diverse.

Mediare le predizioni di più seed è equivalente a un ensemble implicito: riduce la varianza senza aumentare il bias. L'effetto è simile ad addestrare molti più modelli a costo computazionale contenuto.

$$\text{OOF finale} = \frac{1}{|\text{SEEDS}|} \sum_{s \in \text{SEEDS}} \text{OOF}_s$$

---

## 9. Ensemble

### Weighted average

Le predizioni finali sono una media pesata delle probabilità dei tre modelli:

$$p_{ensemble}(x) = w_1 \cdot p_{LGB}(x) + w_2 \cdot p_{XGB}(x) + w_3 \cdot p_{CAT}(x)$$

con $w_1 + w_2 + w_3 = 1$.

### Ricerca dei pesi ottimali

I pesi vengono cercati con grid search sulle predizioni OOF:

```python
for w1 in np.arange(0.10, 0.81, 0.05):
    for w2 in np.arange(0.10, 0.81, 0.05):
        w3 = 1 - w1 - w2
        blend = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
        score = f1_score(y, (blend >= thr).astype(int), average='macro')
```

### Perché l'ensemble funziona

Ogni modello cattura pattern diversi nei dati (diversi bias). Combinandoli, gli errori si "cancellano" parzialmente. Formalmente, se i modelli hanno correlazione $\rho < 1$ tra loro, l'errore dell'ensemble è:

$$\text{Var}(\bar{p}) = \frac{\sigma^2}{n} \cdot [1 + (n-1)\rho]$$

Più bassa è la correlazione tra i modelli, più l'ensemble riduce la varianza.

---

## 10. Soglia di classificazione

Il modello produce probabilità $p \in [0, 1]$. La classe finale si ottiene con una soglia $\tau$:

$$\hat{y} = \begin{cases} 1 & \text{se } p \geq \tau \\ 0 & \text{altrimenti} \end{cases}$$

La soglia di default è 0.5, ma con classi sbilanciate è spesso subottimale. Abbassare la soglia aumenta il recall sulla classe minoritaria a scapito della precision.

### Soglia robusta (CV-based)

Per evitare di ottimizzare la soglia sull'intero OOF (il che introduce un piccolo leakage), il codice usa la **mediana delle soglie ottimali per fold**:

```python
fold_thrs = []
for tr_idx, val_idx in skf_thr.split(blend_oof, y):
    # Trova la soglia ottimale su questo fold di training
    best_local = max([(f1_score(..., thr), thr) for thr in thresholds])
    fold_thrs.append(best_local[1])
robust_thr = np.median(fold_thrs)
```

La mediana è più robusta della media agli outlier (un fold con distribuzione anomala non sposta la soglia finale).

---

## 11. Metrica: F1 Macro

### Precision e Recall

Per la classe positiva (buyers):

$$\text{Precision} = \frac{TP}{TP + FP} \quad \text{(quanti dei predetti come buyer lo sono davvero)}$$

$$\text{Recall} = \frac{TP}{TP + FN} \quad \text{(quanti buyer reali vengono trovati)}$$

### F1 Score

La media armonica di precision e recall:

$$F1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

La media armonica penalizza i valori estremi: un modello con precision 1.0 e recall 0.0 avrà F1 = 0, non 0.5.

### F1 Macro

Calcola l'F1 separatamente per ogni classe, poi fa la media non pesata:

$$F1_{Macro} = \frac{F1_{\text{class 0}} + F1_{\text{class 1}}}{2}$$

A differenza dell'F1 Micro (che pesa per il numero di esempi), l'F1 Macro tratta entrambe le classi allo stesso modo — fondamentale con classi sbilanciate, per non ignorare la minoranza.

---

## Riepilogo delle scelte progettuali

| Componente | Scelta | Motivazione |
|---|---|---|
| Modelli | LGB + XGB + CatBoost | Diversità → riduzione varianza ensemble |
| Gestione sbilanciamento | `scale_pos_weight` | Penalizza di più gli errori sulla classe rara |
| Encoding categoriche | OOF Target Encoding + LabelEncoding | Niente leakage, cattura segnale ordinale |
| Tuning | Optuna TPE | Più efficiente del random search, non serve grid manuale |
| Soglia | CV-based robusta | Evita overfitting sulla soglia |
| Multi-seed | 3 seed per modello | Riduce varianza, equivale a 3× più modelli |
| CV | Stratified 5-fold | Preserva proporzione classi in ogni fold |

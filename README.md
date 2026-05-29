# Steam Game Success Prediction

Predicting whether a game on Steam will be well received by players, using a Wilson-scored success label, K-Means archetypes and three classifiers (Logistic Regression, Random Forest and a tuned XGBoost).

This repository contains the Python notebook that produced every result, figure and table in the report **"Machine Learning Business Problem Solution"**, submitted for the MSc Information Technology Management module *Machine Learning and Visualization for Data* (BSBI).

**Author:** Ibrahim Eren
**Module:** Machine Learning and Visualization for Data, MSc IT Management, BSBI

---

## ⚠️ IMPORTANT — Dataset

> **The dataset is NOT included in this repository.**
>
> The notebook uses the **Steam Games Dataset by FronkonGames** on Kaggle, which contains **122,611 games across 39 columns** (~250 MB CSV file).
>
> - **Source / download:** https://www.kaggle.com/datasets/fronkongames/steam-games-dataset
> - **File needed:** `games.csv`
> - **Why it is not in this repo:** the file is too large for GitHub (the platform recommends keeping files under 100 MB) and the dataset is distributed under Kaggle's own terms, so redistributing the CSV here is not appropriate.
> - **How the notebook loads it:** the first cell mounts Google Drive and reads the CSV from `My Drive/Colab Notebooks/games.csv`. Download the file from Kaggle, place it at that path on your Drive, and the rest of the notebook runs end to end.
> - **Running locally instead?** Replace the Drive mount cell with `df = pd.read_csv('games.csv', low_memory=False)` and keep `games.csv` next to the notebook.
>
> Without this CSV in place, the notebook will not run. Everything else (the cleaned features, archetypes, model code, results) is fully reproduced from this one input file.

---

## What the notebook does

1. Loads the raw Steam Games dataset (122,611 games × 39 columns) from Google Drive.
2. Fixes a column-shift bug in the raw CSV and engineers a clean feature set (`price_real`, `owners_avg`, `release_year`, `platform_count`, `genre_count`, `category_count`, `is_free`, plus one-hot indicators for the top 10 genres).
3. Defines success with the **Wilson lower-bound interval** (at least 10 reviews, Wilson score ≥ 0.80, aligned with Steam's "Very Positive" band). 56,655 games are retained for modelling; 27.5% are successful, 72.5% are not.
4. Runs **K-Means clustering (k = 4)** on log-transformed numeric features to produce four interpretable game archetypes (Proven Classics, New Premium Releases, Cross-Platform, Budget/Casual) and feeds the archetype label back into the classifier as an extra feature.
5. Trains and evaluates **Logistic Regression**, **Random Forest** and a **GridSearchCV-tuned XGBoost** on an 80/20 stratified split, with class-weight balancing for the imbalance.
6. Reports accuracy, precision, recall, F1, ROC-AUC, the confusion matrix and the top 10 feature importances.

**Best model:** tuned XGBoost — Test ROC-AUC **0.757**, F1 **0.547**, CV ROC-AUC 0.750 (no over-fitting).

---

## How to run

The notebook was written and run in **Google Colab** and reads the dataset from Google Drive. This is the easiest way to reproduce the results.

### Option 1 — Google Colab (recommended)

1. Download `games.csv` from the Kaggle link above.
2. Upload it to your Google Drive at: `My Drive/Colab Notebooks/games.csv`
3. Open `steam_success.ipynb` in Colab.
4. Run the first cell — it mounts your Drive and loads the dataset.
5. Run the remaining cells in order.

> Why Google Drive? The Kaggle CSV is ~250 MB, which is too large to commit to GitHub. Mounting Drive is the cleanest way to load it inside Colab without manual uploads each session.

### Option 2 — Run locally (Jupyter)

If you prefer to run locally, replace the Drive mount cell with a direct read from a local path:

```python
df = pd.read_csv('games.csv', low_memory=False)
```

Then install the dependencies and launch Jupyter:

```bash
pip install -r requirements.txt
jupyter notebook steam_success.ipynb
```

---

## Requirements

- Python 3.10+
- pandas, numpy, scipy
- scikit-learn
- xgboost
- matplotlib

See `requirements.txt` for exact versions.

---

## Repository contents

| File | Description |
|---|---|
| `steam_success.ipynb` | The full Colab notebook (data loading → Wilson scoring → clustering → models → evaluation) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |
| `LICENSE` | MIT |

---

## Citation

If referring to this work, please cite the accompanying report and link back to this repository:

> Eren, I. (2026) *Steam game success prediction*. BSBI, MSc Information Technology Management.
> Code: https://github.com/[USERNAME]/[REPO]
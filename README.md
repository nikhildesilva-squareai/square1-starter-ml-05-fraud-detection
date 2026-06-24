# Fraud Detection System — Square 1 AI starter

**Part of [Square 1 AI](https://square1-tutor.vercel.app) · Machine Learning · Project 5.**

✅ **Data included.** The dataset is committed in [`dataset/`](dataset/) and is the **same standardized dataset every learner uses** — so results are comparable. It is 100% synthetic and Square 1-owned (no third-party or personal data). You can also download it as a single file from the project page on Square 1.

To run the commands below, copy the files into `data/` (`mkdir -p data && cp -r dataset/* data/`) or point the commands straight at `dataset/`.

MIT licensed — fork it, build on it, put it in your portfolio.

---

# Fraud Detection System — starter

Starter for Square 1 AI **Machine Learning · Project 5**. Score card transactions for fraud — and serve the model.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data
Download `transactions.csv` from your project page (Resources → Dataset) into `data/`.

## Your task
Three tests define the contract — they fail until you implement the stubs in `fraud/data.py` and `fraud/model.py`:
```bash
pytest -q
python train.py
uvicorn app:app --reload      # then POST a transaction to /score
```
Pipeline: `load_data` → `clean` (fill missing `merchant_category`, normalise `country_code` case, drop the id) → `stratified_split` (preserves the ~1.5% fraud rate) → `train_model` (XGBoost; handle imbalance) → `evaluate` (returns `roc_auc`/`precision`/`recall`; add PR-AUC + confusion matrix yourself) → pick a threshold → serve `POST /score`.

**Accuracy is meaningless at 1.5% fraud** — judge on precision, recall, and PR-AUC, and choose your threshold from the precision/recall trade-off, not the default 0.5. The contract tests run fully offline on tiny inline data (only scikit-learn / numpy / pandas) — `train.py` and `app.py` use XGBoost + FastAPI. Full brief, rubric, and references are on your Square 1 project page. MIT licensed.

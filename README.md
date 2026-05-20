# 📊 Customer Churn Prediction App

A Machine Learning web app built with **Streamlit** that predicts whether a telecom customer will churn, trained on the real **P670 dataset** (5 000 records, 20 features).

---

## 🚀 Live Demo

Deploy to [Streamlit Community Cloud](https://streamlit.io/cloud) for free — see **Deployment** below.

---

## 📁 Project Structure

```
churn-prediction-app/
├── app.py                  # Streamlit application
├── train_model.py          # Full training pipeline → saves .pkl files
├── P670-dataset.xlsx       # Raw dataset (required by train_model.py)
├── churn_model.pkl         # Trained Random Forest  (generated)
├── scaler.pkl              # Fitted MinMaxScaler    (generated)
├── feature_columns.pkl     # Ordered feature list   (generated)
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── .gitignore
└── README.md
```

> The three `.pkl` files are **generated** by running `python train_model.py`.

---

## 🧠 Model & Pipeline

| Step | Detail |
|---|---|
| Raw data | 5 000 rows × 20 columns |
| Outlier removal | IQR-based drop then cap |
| Encoding | `pd.get_dummies` on `state`, `area.code`, `voice.plan`, `intl.plan` |
| Feature engineering | `total_call_minutes`, `total_calls_made`, `total_charge_cost` |
| Scaling | `MinMaxScaler` |
| Algorithm | Random Forest (200 trees, max depth 10) |
| Final features | 76 |
| Test accuracy | **96%** |

---

## ⚙️ Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/churn-prediction-app.git
cd churn-prediction-app
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python train_model.py
# → churn_model.pkl  ✅
# → scaler.pkl       ✅
# → feature_columns.pkl ✅
```

### 5. Run the app
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## ☁️ Deployment — Streamlit Community Cloud

1. Push this repo to GitHub (including the `.pkl` files).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch `main`, main file `app.py`.
4. Click **Deploy**.

---

## 📦 Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
joblib>=1.3.0
openpyxl>=3.1.0
```

---

## 📄 License

[MIT](https://choosealicense.com/licenses/mit/)

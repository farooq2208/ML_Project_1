# Sleep Health Predictor

A machine-learning–powered web application that predicts sleep disorders from
everyday lifestyle and health metrics.

---

## Project Description

Poor sleep is linked to dozens of serious health conditions, yet millions of
people are unaware they may have a diagnosable sleep disorder.  
**Sleep Health Predictor** gives users an instant, AI-driven assessment of
their sleep health — identifying whether they are likely to be *Healthy*,
experiencing *Insomnia*, or showing signs of *Sleep Apnea* — based on 12
simple inputs such as age, occupation, stress level, and blood pressure.

---

## About the Machine-Learning Model

| Property | Detail |
|---|---|
| **Algorithm** | XGBoost Classifier |
| **Selection method** | 5-fold GridSearchCV across Logistic Regression, Decision Tree, Random Forest, and XGBoost |
| **Optimisation metric** | Weighted F1-Score (chosen because the classes are imbalanced) |
| **Preprocessing** | StandardScaler for numerics · OneHotEncoder for nominal categories · OrdinalEncoder for BMI |
| **Target classes** | Healthy (0 · no disorder) · Insomnia (1) · Sleep Apnea (2) |
| **Dataset** | Sleep Health & Lifestyle dataset — 374 records, 13 features |

The full training pipeline is preserved in `sleep_disorder_model.pkl` (a
scikit-learn `Pipeline` object) so that the Streamlit app can pass raw,
unscaled inputs directly to the model and receive predictions without any
manual preprocessing.

---

## Application Features

- **Intuitive input form** — sliders, dropdowns, and number fields with
  helpful tooltips for every parameter.
- **Input validation** — catches impossible values (e.g. diastolic ≥ systolic)
  before the model is called.
- **Confidence breakdown** — colour-coded probability bars for each class.
- **Input summary expander** — review submitted values alongside the result.
- **Sidebar** — project overview, how-to-use instructions, and model details
  always visible.
- **Production-ready** — cached model loading, clean error handling, and zero
  external API calls.

---

## Project Structure

```
sleep-health-predictor/
│
├── app.py                    # Streamlit application (main entry point)
├── sleep_disorder_model.pkl  # Pre-trained XGBoost pipeline
├── sleep_health.csv          # Original training dataset (reference only)
├── Sleep_health.ipynb        # Model-training notebook (reference only)
├── requirements.txt          # Pinned Python dependencies
└── README.md                 # This file
```

> **Note:** `sleep_health.csv` and `Sleep_health.ipynb` are not required at
> runtime. Only `app.py`, `sleep_disorder_model.pkl`, and `requirements.txt`
> are needed for deployment.

--

## Installation & Local Setup

### Prerequisites

- Python **3.11** or **3.12** (recommended)
- `pip` (bundled with Python)
- `git` (optional, for cloning)

### 1 — Clone or download the repository

```bash
git clone https://github.com/your-username/sleep-health-predictor.git
cd sleep-health-predictor
```

Or simply place `app.py`, `sleep_disorder_model.pkl`, and `requirements.txt`
in the same folder.

### 2 — Create a virtual environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate.bat

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4 — Run the application

```bash
streamlit run app.py
```

Streamlit will open a browser tab at `http://localhost:8501` automatically.

---

## How to Use the Application

1. **Fill in the form** — the three-column panel covers Demographics, Sleep &
   Activity, and Health Vitals.
2. **Click "Predict Sleep Health"** — the model runs instantly.
3. **Read your result** — a colour-coded card shows the predicted class with
   an explanation.
4. **Check the confidence bars** — see the probability score for each class.
5. **Expand "View your submitted inputs"** — verify the values used for
   prediction.

---

## Assumptions & Limitations

| Item | Detail |
|---|---|
| **Age range** | The model was trained on data for ages 27–59. Predictions for younger or older users are extrapolations. |
| **Occupation list** | Only the 11 occupations present in the training data are available. Choose the closest match if yours is not listed. |
| **Blood pressure** | The model uses systolic and diastolic values separately; ensure you enter them correctly. |
| **Not medical advice** | This tool provides an informational estimate only. It cannot replace a clinical diagnosis. |
| **Dataset size** | 374 training records is small; real-world performance may vary. |
| **Model version** | The pickle file was created with scikit-learn 1.9.0 and XGBoost 3.3.0. Use matching versions to avoid compatibility warnings. |

---

## Technologies & Libraries

| Library | Version | Purpose |
|---|---|---|
| [Streamlit](https://streamlit.io) | 1.46.0 | Web application framework |
| [scikit-learn](https://scikit-learn.org) | 1.9.0 | Preprocessing pipeline & model utilities |
| [XGBoost](https://xgboost.readthedocs.io) | 3.3.0 | Gradient-boosting classifier |
| [pandas](https://pandas.pydata.org) | 3.0.2 | DataFrame construction for model input |
| [NumPy](https://numpy.org) | 2.4.4 | Numerical operations |
| [joblib](https://joblib.readthedocs.io) | 1.5.3 | Model serialisation / deserialisation |
| [PyArrow](https://arrow.apache.org/docs/python/) | 20.0.0 | pandas string-backed dtype backend |

---

## Live Demo

**Live Demo: https://sleep-health-predictor2208.streamlit.app/**

---

## Deploying to Streamlit Community Cloud

1. Push your project folder (with `app.py`, `sleep_disorder_model.pkl`, and
   `requirements.txt`) to a **public GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **New app**, select your repository and branch, and set the main
   file path to `app.py`.
4. Click **Deploy** — Streamlit installs the requirements automatically.
5. Once live, copy the URL and update the **Live Demo** link above.

---

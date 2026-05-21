# Predictive Maintenance & Manufacturing KPI Analytics System

An **Industry 4.0 smart factory simulation** built to demonstrate how data science and machine learning can transform maintenance from reactive firefighting to proactive optimization.

This portfolio project combines **manufacturing KPIs** with **failure prediction intelligence** to reduce unexpected downtime, improve asset utilization, and support continuous process improvement.

---

## Business Value

| Traditional (Reactive) | Predictive (This Project) |
|---|---|
| Repairs happen after breakdown | Failures are anticipated before they occur |
| High unplanned downtime | Lower downtime through early intervention |
| Maintenance schedules are generic | Maintenance actions are risk-based and targeted |
| Production losses are hard to explain | KPI + root-cause signals support operational decisions |

### Key outcomes this system targets
- Reduced unplanned machine stoppages.
- Better planning for maintenance teams and spare parts.
- Increased production stability and quality throughput.
- Improved operational transparency for leadership reporting.

---

## Industry 4.0 Architecture Flow

```text
Sensor Streams / Historical Machine Logs
            │
            ▼
   Data Ingestion & Cleaning (utils/data_prep.py)
            │
            ▼
 Feature Engineering + KPI Layer (OEE, MTBF, MTTR)
            │
            ▼
 XGBoost Failure Model Training (models/train_model.py)
            │
            ▼
      Serialized Artifacts (.pkl via joblib)
            │
            ▼
 Streamlit Multi-Page Dashboard (app.py)
   ├─ Factory Overview
   ├─ KPI Analytics
   ├─ Live Prediction
   └─ Alerting + Root Cause Rules
```

---

## Project Structure

```bash
.
├── app.py
├── data/
├── notebooks/
├── models/
│   └── train_model.py
├── dashboard/
├── reports/
├── utils/
│   └── data_prep.py
├── requirements.txt
└── README.md
```

### Exact terminal commands to create the folders
```bash
mkdir -p data notebooks models dashboard utils reports
```

---

## KPIs Tracked

| KPI | Description | Why It Matters |
|---|---|---|
| **OEE (Overall Equipment Effectiveness)** | Availability × Performance × Quality | Global efficiency benchmark for machine productivity |
| **Availability** | Operating Time / Planned Production Time | Captures downtime impact |
| **Performance** | Actual Throughput vs Ideal Throughput | Measures speed losses |
| **Quality** | Good Units / Total Units | Measures quality losses |
| **MTBF** | Mean Time Between Failures | Reliability indicator |
| **MTTR** | Mean Time To Repair | Maintainability / service efficiency indicator |

---

## Machine Learning Model

- **Algorithm:** XGBoost Classifier
- **Task:** Binary classification (`machine_failure` = 1 or 0)
- **Feature engineering includes:**
  - Moving averages for temperature and rotational speed.
  - Temperature gradient and gradient change.
  - Composite **Machine Health Score** from normalized stress factors.
- **Evaluation emphasis:** high **Recall** and reduced **False Negatives**, with Precision, F1, and ROC-AUC reported.

---

## Setup & Run Locally

### 1) Clone repository
```bash
git clone <your-repo-url>
cd Manufacturing-Analysis
```

### 2) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Add the Kaggle AI4I dataset
Place the CSV in:
```bash
data/ai4i2020.csv
```

### 5) Prepare data
```bash
python utils/data_prep.py
```

### 6) Train model
```bash
python models/train_model.py
```

### 7) Launch dashboard
```bash
streamlit run app.py
```

---

## Dashboard Pages

1. **Factory Overview**
   - Machine status, high-level OEE, uptime.
2. **KPI Analytics**
   - Downtime trend analysis and process-variable relationships.
3. **Predictive Maintenance (Live)**
   - Real-time failure probability and machine health score from user inputs.
4. **Alerting & Root Cause**
   - High-risk warnings and rules-based probable cause guidance.

---

## Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Plotly
- Joblib

---

## Recruiter Note

This project is intentionally designed to mirror industrial analytics workflows relevant to **BMW**, **Siemens**, **Bosch**, and other advanced manufacturing environments by combining:
- reliability engineering KPIs,
- interpretable operational dashboards,
- and machine-learning-driven predictive maintenance.

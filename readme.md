# 📦 DataCo Supply Chain: AI Delivery Risk Predictor

<div align="center">

![VS Code](https://img.shields.io/badge/VS_Code-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)

### _An end-to-end Machine Learning web application predicting supply chain delays before they happen._

**Can we predict late deliveries? | Why do they happen? | How do we intervene proactively?**

**I built an AI tool to transition supply chain management from reactive to proactive.**

<div align="center">

[🚀 Live App Demo](https://ai-delivery-risk-predictor-dataco.streamlit.app) • [💻 Explore Code](app.py) • [🤝 Connect on LinkedIn](https://www.linkedin.com/in/anshul-silhare)


</div>

---

## 🎯 Project Motivation

As a PGDM student in Research & Business Analytics at WeSchool, my previous projects focused heavily on descriptive analytics (SQL extraction, Power BI dashboards). While historical tracking is great for analyzing past performance, **it is fundamentally reactive.**

Stakeholders don't just want to know _why_ an order was late last month; they want to know if _tomorrow's_ order is going to be late so they can intervene today.

**The Goal:** Build a predictive intelligence tool that flags high-risk deliveries based on real-time factors like shipping mode, region, scheduled days, and order quantity, wrapping complex Machine Learning in a simple, business-friendly UI.

---

## 📊 Dataset Overview

**Source:** DataCo Smart Supply Chain Dataset
**Scale:** 180,000+ global shipping records
**Business Domain:** Supply Chain, Logistics, Operations
**Target Variable:** Late Delivery Risk (Binary Classification: 1 = Late, 0 = On Time)

**Data Preprocessing Highlights:**

- Cleansed missing values and handled class imbalances.
- Applied **One-Hot Encoding** to categorical variables (Order Region, Shipping Mode).
- Utilized `StandardScaler` to normalize numerical variances (Quantity, Days Scheduled).
- Resulted in a complex training matrix of **240+ independent features**.

---

## ✨ Key Features (The UI)

### 1️⃣ Real-Time Risk Prediction

Users can input order details (Shipping Mode, Region, Quantity, Scheduled Days) into a clean, dark-mode Streamlit interface. The AI instantly processes the input and returns a **Probability Risk Score** (e.g., "85% certain this order will be delayed").

### 2️⃣ Explainable AI (Visual Insights)

Business users need to know _why_ the AI made its decision. The dashboard utilizes the Random Forest's `feature_importances_` to display dynamic visual charts showing exactly which factors drove the risk score up or down.

### 3️⃣ Bulk Order Processing

Supply chain managers don't check orders one by one. The app includes a sidebar where users can upload a CSV of 500+ unfulfilled orders. The AI analyzes the entire batch and outputs a downloadable CSV with predicted delay risks appended to every row.

---

## 🧠 Under the Hood: Engineering Challenges Solved

Building the model in a Jupyter Notebook was the easy part. Deploying it to a live web application presented significant engineering hurdles:

### Challenge 1: Matrix Dimensionality Mismatch

**The Problem:** The AI was trained on 240+ columns (due to One-Hot Encoding regions and shipping modes), but the Streamlit UI only asks the user for 4 simple inputs.
**The Fix:** I engineered a dynamic routing script that leverages `model.feature_names_in_`. When a user submits an order, the code generates a zero-filled skeleton matrix of all 240 columns, maps the specific user inputs to the correct One-Hot index, and perfectly aligns the shape before feeding it to the AI.

### Challenge 2: The Self-Healing Scaler

**The Problem:** `StandardScaler` crashes if you feed it One-Hot Encoded columns it hasn't seen, or if the columns are in the wrong order.
**The Fix:** Built logic to specifically isolate numerical columns (`scaler.feature_names_in_`) from the 240-column matrix, scale _only_ those numbers, and stitch the dataframe back together in milliseconds.

### Challenge 3: Environment "Dependency Hell"

**The Problem:** Deep low-level C++ DLL load failures caused by version clashing between pip-installed PyArrow and Conda-installed Scikit-Learn in Python 3.13.
**The Fix:** Architected a strict `conda-forge` environment, completely aligning the underlying C++ binaries and ensuring 100% stability across Pandas, NumPy, and Scikit-Learn.

---

### 🤖 Modern Development Workflow (Vibecoding)

While the core Machine Learning model, data engineering, and matrix-alignment logic were built manually, I leveraged **Google Antigravity (AI-assisted Agent IDE)** to rapidly prototype and generate the front-end Streamlit UI.

By directing the AI agent to implement the visual components (Plotly charts, Lottie animations, and layout), I was able to focus my engineering time strictly on solving the complex backend mathematical and environmental architecture. This hybrid approach allowed me to deploy a polished, enterprise-grade product in a fraction of the traditional development time.

## 🛠️ Technical Stack

<div align="center">

| **Category**                 | **Technologies**                         |
| ---------------------------- | ---------------------------------------- |
| **Language**                 | Python 3.13                              |
| **Machine Learning**         | Scikit-Learn (Random Forest Classifier)  |
| **Data Processing**          | pandas, numpy, joblib                    |
| **Front-End / UI**           | Streamlit, Plotly, Streamlit-Lottie      |
| **Environment Architecture** | Anaconda (conda-forge)                   |
| **AI Development Tools**     | Google Antigravity (only for Web App UI) |

</div>

---

## 🚀 Getting Started

Want to run the AI on your local machine? Due to the complex Scikit-Learn dependencies, using Conda is highly recommended.

### Prerequisites

- Anaconda or Miniconda installed
- Git

### Installation & Setup

**1. Clone this repository:**

```bash
git clone [https://github.com/AnshulSilhare/dataco-supply-chain-ai.git](https://github.com/AnshulSilhare/dataco-supply-chain-ai.git)
cd dataco-supply-chain-ai
```

**2. Create the exact environment (Crucial for DLL stability):**

```bash
conda create -n dataco_env python=3.13 -y
conda activate dataco_env
```

**3. Install dependencies via Conda-Forge:**

```bash
conda install -c conda-forge scikit-learn numpy pandas pyarrow streamlit joblib -y
```

**4. Launch the Application:**

```bash
streamlit run app.py
```

(The web app will automatically open in your default browser at http://localhost:8501)

## 📁 Repository Structure

```bash
📦 dataco-supply-chain-ai
│
├── 📂 notebooks/
│   └── Model_Training.ipynb       # Data cleaning, EDA, and model training logic
│
├── 📄 app.py                      # Main Streamlit web application code
├── 📄 dataco_rf_model.joblib      # Saved Random Forest AI Brain
├── 📄 dataco_scaler.joblib        # Saved Numerical Scaler
├── 📄 dataco_columns.joblib       # Saved 240+ Matrix Architecture
├── 📄 requirements.txt            # Dependency list
└── 📄 README.md                   # This file
```

## 🔮 Future Roadmap

Phase 2: Advanced Integrations

[ ] Connect Streamlit directly to a live SQL database for real-time order ingestion.
[ ] Add Geospatial visualization (Plotly Maps) to map high-risk shipping routes.
[ ] Implement XGBoost alongside Random Forest to create an ensemble voting mechanism for higher accuracy.

📫 Let's Connect

<div align="center">

</div>

I'm actively seeking Summer 2026 roles in Business Analytics or Data Analytics.

If you are looking for an analyst who can not only build SQL queries and Power BI dashboards, but also engineer and deploy production-ready Machine Learning tools, let's connect!

<div align="center">

⭐ Support This Project
If you found this supply chain architecture helpful, please consider leaving a star!

Built with 🐍 Python | 📦 Machine Learning | 💡 Operations Insights

Developed by Anshul Silhare PGDM Student | Research & Business Analytics | WeSchool March 2026

</div>

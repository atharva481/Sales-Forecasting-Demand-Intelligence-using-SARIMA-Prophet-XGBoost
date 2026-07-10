# 📈 Sales Forecasting & Demand Intelligence using SARIMA, Prophet, and XGBoost

## Overview

This project develops an end-to-end **Demand Forecasting and Inventory Intelligence System** that predicts future sales, detects demand anomalies, and generates inventory recommendations using machine learning and time-series forecasting techniques.

The project compares three forecasting models—**SARIMA**, **Facebook Prophet**, and **XGBoost**—to determine the most accurate approach for sales prediction while providing actionable business insights for inventory optimization.

---

## Key Features

- 📊 Time-series sales forecasting
- 🤖 Machine Learning forecasting using XGBoost
- 📈 Statistical forecasting using SARIMA
- 🔮 Trend and seasonality forecasting using Prophet
- 📉 Stationarity analysis using Augmented Dickey-Fuller (ADF) Test
- 🚨 Demand anomaly detection using Isolation Forest & Z-Score
- 📦 Inventory stocking recommendations
- 📊 Automatic model comparison using MAE, RMSE and MAPE

---

## Problem Statement

Accurate sales forecasting is essential for:

- Inventory optimization
- Supply chain planning
- Demand prediction
- Revenue forecasting
- Reducing stock-outs
- Minimizing overstock costs

This project builds an intelligent forecasting pipeline capable of selecting the best-performing forecasting model while generating practical inventory decisions.

---

# Workflow

```
Historical Sales Data
          │
          ▼
Data Cleaning & Preprocessing
          │
          ▼
Stationarity Testing (ADF)
          │
          ▼
Feature Engineering
          │
          ▼
 ┌────────────┬──────────────┬─────────────┐
 │            │              │
 ▼            ▼              ▼
SARIMA     Prophet       XGBoost
 │            │              │
 └────────────┴──────────────┘
              │
              ▼
 Model Evaluation
(MAE • RMSE • MAPE)
              │
              ▼
 Best Model Selection
              │
              ▼
Future Sales Forecast
              │
              ▼
Demand Intelligence
              │
              ▼
Inventory Recommendations
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Statsmodels
- Prophet
- SciPy
- Isolation Forest

---

# Machine Learning Models

## 1. SARIMA

Seasonal AutoRegressive Integrated Moving Average (SARIMA) captures:

- Trend
- Seasonality
- Autocorrelation

Best suited for traditional statistical forecasting.

---

## 2. Prophet

Facebook Prophet models:

- Trend
- Seasonality
- Holiday effects

Useful for interpretable forecasting with minimal tuning.

---

## 3. XGBoost

Gradient Boosting model using engineered lag and rolling-window features.

Advantages:

- Captures nonlinear demand patterns
- Handles promotions and external variables
- Fast training
- High forecasting accuracy

---

# Stationarity Analysis

The Augmented Dickey-Fuller (ADF) Test was used to verify stationarity.

### Original Series

| Metric | Value |
|---------|--------|
| ADF Statistic | -4.4161 |
| p-value | 0.0003 |

**Result:** Stationary ✅

---

### After First-Order Differencing

| Metric | Value |
|---------|--------|
| ADF Statistic | -8.7271 |
| p-value | 0.000000 |

**Result:** Stationary ✅

---

# Model Performance

| Model | MAE | RMSE | MAPE |
|--------|---------:|---------:|---------:|
| **XGBoost** | **$38,523** | $45,604 | **9.7%** |
| SARIMA | $39,509 | **$43,230** | 10.3% |
| Prophet | $47,105 | $52,909 | 14.5% |

## Best Performing Model

🏆 **XGBoost**

Reason:

- Lowest MAE
- Lowest MAPE
- Handles nonlinear demand patterns
- Supports additional business features
- No stationarity requirement

---

# Demand Forecast

Example Forecast

| Month | Predicted Sales |
|---------|---------------:|
| Month +1 | $262,234 |
| Month +2 | $366,376 |
| Month +3 | $343,473 |

---

# Anomaly Detection

Two anomaly detection techniques were compared.

| Method | Anomalies Detected |
|----------|------------------:|
| Isolation Forest | 11 |
| Z-Score | 0 |

Isolation Forest successfully detected unusual demand patterns that traditional statistical methods failed to identify.

---

# Inventory Intelligence

Based on clustering analysis, products were categorized into four inventory strategies.

### High Volume, Stable

- Maintain 15% safety stock
- Automatic replenishment
- Prevent stock-outs

---

### Growing Demand

- Increase procurement by 20–25%
- Secure supplier contracts
- Monitor demand growth

---

### Low Volume, Volatile

- Just-In-Time ordering
- Minimal inventory holding
- Quarterly review

---

### Declining Products

- Reduce reorder quantity
- Clearance promotions
- Consider product delisting

---

# Repository Structure

```
Sales-Forecasting-Demand-Intelligence/
│
├── data/
├── notebooks/
├── charts/
├── models/
├── outputs/
├── requirements.txt
├── README.md
└── main.ipynb
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/atharva481/Sales-Forecasting-Demand-Intelligence-using-SARIMA-Prophet-XGBoost.git

cd Sales-Forecasting-Demand-Intelligence-using-SARIMA-Prophet-XGBoost
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
jupyter notebook
```

or

```bash
python main.py
```

---

# Business Impact

This project demonstrates how forecasting models can improve:

- Demand Planning
- Supply Chain Management
- Inventory Optimization
- Revenue Forecasting
- Business Decision Making

---

# Future Enhancements

- Deep Learning (LSTM/GRU)
- Transformer-based forecasting
- Real-time forecasting dashboard
- Weather and holiday integration
- Automated retraining pipeline
- Cloud deployment (AWS/Azure/GCP)

---

# Results

✅ Sales forecasting using three forecasting models

✅ Time-series stationarity analysis

✅ Automatic model comparison

✅ Demand anomaly detection

✅ Inventory optimization recommendations

✅ Business-oriented forecasting pipeline

---

# Author

**Atharva Sawant**

**LinkedIn:** *(Add your profile link)*

**GitHub:** https://github.com/atharva481

---

## License

This project is licensed under the MIT License.

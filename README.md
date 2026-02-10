# ⚡ EV Predictive Maintenance & Battery Intelligence System 🚗🔋

An AI-powered predictive maintenance and battery monitoring system for **Electric Vehicles (EVs)** that analyzes battery behavior, predicts remaining driving range, detects abnormalities, and provides AI-based advisory insights.

---

## 📌 Project Overview

Electric Vehicles rely heavily on battery health and efficient energy usage.  
This project aims to provide an **intelligent monitoring and predictive system** that helps:

- 🔍 Detect abnormal battery discharge patterns  
- 📉 Monitor battery State of Charge (SOC)  
- 📏 Predict remaining driving range  
- 🧠 Provide AI advisory insights for maintenance  
- 📊 Display model accuracy and prediction confidence  

The system combines **Machine Learning**, **Data Analysis**, and **AI advisory logic** to support smarter EV battery management.

---

## ✨ Key Features

✅ **Battery Health Monitoring**  
Tracks battery-related parameters and detects unusual discharge behavior.

✅ **SOC Prediction**  
Uses trained ML models to estimate the State of Charge accurately.

✅ **Driving Range Prediction**  
Predicts how many kilometers can be traveled with the current battery condition.

✅ **Predictive Maintenance Alerts**  
Identifies potential battery degradation or service requirements early.

✅ **AI Advisory Report**  
Provides explainable AI-based suggestions after analysis.

✅ **Model Accuracy Display**  
Shows validated model performance metrics (e.g., ≈90% accuracy).

✅ **Prediction Confidence Indicator**  
Displays confidence score **after analysis** (NIL before prediction).

---

## 🧠 Machine Learning Details

- **Problem Type:** Regression  
- **Models Used:**  
  - SOC Prediction Model  
  - Range Prediction Model  
- **Evaluation Metrics:**  
  - R² Score  
  - Mean Absolute Error (MAE)  

> ⚠️ Since this is a regression-based system, prediction probabilities are not directly available.  
> The displayed accuracy represents **offline validation performance**, not per-sample probability.

---

## 📊 Accuracy & Confidence Explanation

| Metric | Description |
|------|------------|
Model Accuracy | Fixed value derived from offline validation (≈90%) |
Prediction Confidence | Calculated only after prediction |
Initial State | Displays **NIL** until analysis is performed |

---

## 🏗️ System Architecture

1. **Input Parameters**  
   - Battery SOC  
   - Voltage / Current (derived)  
   - Distance requirement  

2. **ML Model Inference**  
   - SOC estimation  
   - Range prediction  

3. **AI Advisory Engine**  
   - Maintenance suggestion  
   - Risk indicators  

4. **UI Dashboard Output**  
   - Predictions  
   - Accuracy  
   - Confidence score  

---

EV-Predictive-Maintenance-AI/
│
├── models/
│ ├── soc_model.pkl
│ ├── range_model.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── assets/


---

## 🔗 Pretrained Models (Hosted on Hugging Face)

- SOC Model  
  👉 https://huggingface.co/mahez/EV-Predictive-Maintenance-AI/resolve/main/soc_model.pkl

- Range Prediction Model  
  👉 https://huggingface.co/mahez/EV-Predictive-Maintenance-AI

---

## 🚀 How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Mahez-snow/EV-Predictive-Maintenance-AI.git
cd EV-Predictive-Maintenance-AI


## 📁 Project Structure


EV AI Smart Mission & Health Dashboard
An integrated Predictive Maintenance and Mission Control system for Electric Vehicles (EVs). This project leverages Machine Learning to transform raw IoT sensor data into actionable insights, helping drivers mitigate "Range Anxiety" and manage battery longevity.

📌 Project Overview
This project addresses the critical need for intelligence in the EV industry by analyzing high-frequency sensor data to predict vehicle performance and maintenance requirements.

🛠️ Key Technical Features
Precision Battery Analytics (SoC & SoH): Utilizes regression models to monitor the State of Charge (current energy) and State of Health (long-term degradation) by analyzing voltage, cycles, and thermal stress.

Proactive Fault Detection: A classification system that identifies Abnormal Discharge patterns—detecting energy leaks or thermal anomalies where current draw is disproportionate to vehicle speed.

Dynamic Mission Controller: An AI "co-pilot" that estimates Remaining Range based on vehicle load and road conditions, providing an Optimal Speed Recommendation to ensure the destination is reached safely.

💻 Tech Stack
Language: Python

Machine Learning: Scikit-Learn (Random Forest Regressor & Classifier)

Web Framework: Streamlit (Custom CSS Dashboard)

Data Handling: Pandas, Numpy

Model Serialization: Joblib

📁 Repository Structure
├── app.py                     # Main Streamlit application
├── EV_Project.ipynb           # Training notebook with deep data analysis
├── requirements.txt           # List of necessary Python libraries
├── soc_model.pkl              # Pre-trained SoC Model
├── health_model.pkl           # Pre-trained SoH Model
├── range_model.pkl            # Pre-trained Range Model
├── abnormal_discharge_model.pkl # Pre-trained Fault Detector
├── speed_recommendation_model.pkl # Pre-trained Efficiency Model
├── low_battery_model.pkl      # Pre-trained Alert Model
└── logo.png                   # Dashboard UI assets

🧪 Methodology & Logic

The system was trained on a comprehensive IoT-based EV Dataset.
      Absolute Current Logic: The AI monitors Current_Mag (magnitude) to handle both charging and discharging states (positive/negative current) accurately.
      Threshold-Based Faults: Abnormal Discharge is triggered when high current magnitudes (>120A) occur at relatively low speeds (<45km/h) or if internal temperatures exceed 60°C.
      Mission Physics: Range is dynamically calculated using the formula:  Range = (SoC \times 400) - (Weight \times 0.04) - (Roughness \times 20)

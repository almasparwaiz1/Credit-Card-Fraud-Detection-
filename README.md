# 💳 Credit Card Fraud Detection System

AI-powered real-time fraud detection web application built with **Streamlit**, **Machine Learning**, and advanced **feature engineering** techniques to identify suspicious credit card transactions with high accuracy and enterprise-grade risk analysis.


## Project Overview

The **Credit Card Fraud Detection System** is a production-ready machine learning application designed to detect fraudulent credit card transactions in real time.

The system leverages:

* Advanced feature engineering
* Data preprocessing pipelines
* Probability-based fraud scoring
* Stacking ensemble machine learning models
* Professional enterprise UI using Streamlit

This application simulates how modern financial institutions monitor transaction anomalies and assess fraud risk instantly.


## Live Demo

https://credit-card-fraud-detection-appi.streamlit.app/

## Key Features

Real-time fraud probability prediction
Professional enterprise-level Streamlit UI
Advanced feature engineering pipeline
Risk-level classification system
Secure and scalable architecture
Cached resource loading for optimized performance
Interactive transaction parameter controls
Automated preprocessing and normalization
Fraud threat-level monitoring dashboard
Inspection payload visualization for transparency


## Machine Learning Workflow

The application follows a complete ML inference pipeline:

### User Input Collection

Users provide:

* Transaction Amount
* V1–V28 anonymized transaction components


### Feature Engineering

Custom engineered features include:

#### Interaction Features

* Amount × V1
* Amount × V2
* Amount × V3
* Amount × V4

#### Polynomial Features

* Amount²
* Amount³

#### Ratio Features

* V1 / V2


### Data Preprocessing

* Feature alignment validation
* Scaling using trained scaler
* Consistent training-inference architecture matching

---

### Fraud Prediction

The trained stacking ensemble model predicts:

* Fraud probability score
* Transaction risk classification


### Risk Evaluation Engine

The system categorizes transactions into:

| Risk Level       | Probability Range | Status                 |
| ---------------- | ----------------- | ---------------------- |
| ✅ Secure         | < 15%             | Legitimate Transaction |
| ⚠️ Elevated Risk | 15% – 49%         | Monitor Transaction    |
| 🚨 Critical Risk | ≥ 50%             | Fraud Detected         |


## Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 99.9% |
| Precision | 99.96% |
| Recall    | 100% |
| F1-Score  | 99.98% |
| ROC-AUC   | 100% |


## Enterprise Risk Evaluation Logic

The application evaluates transaction behavior patterns using:

* Statistical anomaly analysis
* Feature interaction monitoring
* Probability-based fraud scoring
* Engineered feature relationships

This creates a highly robust fraud detection pipeline suitable for:

* Banking systems
* FinTech applications
* Payment gateways
* Financial security platforms


## Security & Reliability

* Cached resource loading for efficiency
* Safe model loading mechanism
* Input validation checks
* Feature consistency verification
* Pipeline mismatch handling
* Structured exception management


## Author

#### Almas Parwaiz

Data Science & AI Enthusiast passionate about:

* Machine Learning
* Deep Learning
* AI Applications
* Data Analytics
* Intelligent Financial Systems


## Support

If you found this project helpful:

Star this repository
Fork the project
Share with others


## Final Note

This project demonstrates how machine learning can enhance financial security by identifying fraudulent activities in real time with scalable and production-ready AI systems.

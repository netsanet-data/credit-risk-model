# Credit Risk Probability Model for Alternative Data

## Project Overview

This project develops a credit risk probability model using alternative transaction data from the Xente dataset. The objective is to predict customer creditworthiness in situations where traditional credit history is unavailable.

## Credit Scoring Business Understanding

### Basel II and Risk Measurement

Basel II emphasizes accurate risk measurement and transparent risk management practices within financial institutions. For Bati Bank, this means that any credit scoring model must be interpretable, explainable, and well documented. Financial regulators require institutions to justify lending decisions and demonstrate how risk assessments are performed. Therefore, model transparency is just as important as predictive performance.

### Why a Proxy Target Variable is Necessary

The Xente dataset does not contain a direct loan default label. Because no explicit default outcome exists, a proxy target variable must be created using customer transaction behavior. Techniques such as RFM (Recency, Frequency, Monetary) analysis can identify high-risk and low-risk customer groups.

Using a proxy target introduces business risks. The proxy may not perfectly represent actual loan default behavior, potentially causing incorrect risk classifications. As a result, continuous monitoring and validation are required before deployment.

### Interpretable vs High-Performance Models

There is an important trade-off between model interpretability and predictive performance.

Interpretable models such as Logistic Regression combined with Weight of Evidence (WoE) transformations provide clear explanations for credit decisions and support regulatory compliance.

More complex models such as Gradient Boosting or XGBoost often achieve higher predictive accuracy but may be harder to explain to regulators and business stakeholders.

In a regulated banking environment, achieving the right balance between explainability and predictive performance is critical for successful credit risk management.

## Dataset

Source: Xente Fraud Detection Dataset

Files:

* data.csv
* Xente_Variable_Definitions.csv

## Project Structure

* notebooks/: exploratory analysis
* src/: reusable source code
* tests/: unit tests
* data/: dataset files

## Setup

pip install -r requirements.txt

## Author

Netsanet Gebrekidan


# Explainable AI: GAM-Based Transparent Models & Post-hoc Methods

Overview

This repository contains implementations of GAM-based transparent models, including Neural Additive Models (NAM) and Neural Basis Models (NBM). These models have been applied to various financial (e.g., Loan Default Prediction) and non financial datasets, demonstrating their effectiveness through performance evaluations and visualization plots.

Additionally, this project explores post-hoc explanation methods, such as SHAP (SHapley Additive Explanations) and LIME (Local Interpretable Model-agnostic Explanations). Simple sentiment classification experiments were conducted to illustrate their functionality, while further analysis was performed on tabular datasets to assess their sensitivity to noise. Through systematic experimentation, this repository provides insights into how SHAP and LIME react to perturbations in input data.

## Model Performance Comparison

Test-set performance (mean ± std). For regression tasks (Abalone, CA Housing, Wine), lower is better (RMSE). 
For classification tasks (Credit, Churn, Telescope), higher is better (AUROC). 
Bold numbers indicate the best result in each column.

![Performance Plot](../plot/insurance.png)


# Explainable AI: GAM-Based Transparent Models & Post-hoc Methods

## Description

As machine learning is increasingly applied to critical domains like finance and healthcare, the demand for models that not only perform well but also provide clear and trustworthy explanations has become essential. In high-stakes environments, black-box models pose challenges in terms of accountability, regulatory compliance, and user trust.

Despite advances in explainable AI, there is still no universally superior method that balances interpretability and predictive power. This research aims to bridge that gap by exploring both intrinsically interpretable models and post-hoc explanation techniques.

This repository presents implementations of GAM-based transparent models such as Neural Additive Models (NAM) and Neural Basis Models (NBM). These models are designed to retain interpretability while handling complex data relationships, and are tested on real-world financial datasets like loan default prediction, insurance claim classification, fraud detection, and customer churn prediction.

In addition, post-hoc methods like SHAP and LIME are examined through controlled experiments on both textual and tabular data. Their sensitivity to noise and perturbations is analyzed to better understand the reliability of their explanations.

Through systematic comparison and visualization, this project contributes to the practical understanding of how interpretable models and explanation techniques can be applied to real-world problems, especially in domains where decisions must be transparent, accountable, and explainable.

## Model Performance Comparison

Test-set performance (mean ± std). This table summarizes the AUROC scores of various interpretable and black-box models across five binary classification tasks. Among them, three tasks involve insurance claim prediction (Omdema, Car, Allstate), one task is for fraud detection (Fraud), and one task for customer churn prediction (Churn). Higher is better, and bold values indicate the best performance per dataset.

![Performance Plot](../plot/insurance.png)

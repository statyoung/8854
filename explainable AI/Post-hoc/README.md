# Model Explanation Examples with LIME and SHAP

This repository provides hands-on examples of using **LIME** and **SHAP**, two popular post-hoc explainable AI (XAI) techniques. The notebooks demonstrate how to apply these methods to both tabular and text data, and compare their explanation characteristics.

## Contents

### 1. `LIME-tabular.ipynb`
**Explaining Tabular Data with LIME**  
Applies LIME to a classification model trained on structured/tabular data. It visualizes local explanations and highlights the features driving each prediction.

### 2. `LIME-wordvec.ipynb`
**Text Model Interpretation using LIME with Word Embeddings**  
Demonstrates how LIME works with models that take word vector inputs (e.g., using Word2Vec or embedding layers). Useful for interpreting text classification models.

### 3. `SHAP.ipynb`
**SHAP Explanations for Tabular Data**  
Uses SHAP values to provide a global and local interpretation of a model trained on structured data. Includes summary plots and force plots to show feature impacts.

### 4. `SHAP-wordvec.ipynb`
**Explaining Text Classifiers with SHAP**  
Applies SHAP to models using word embeddings as inputs. Shows how SHAP can be adapted to NLP use cases, including token-level contribution visualization.

### 5. `SHAP_LIME_stdcomparison.ipynb`
**Comparing SHAP and LIME: A Stability Perspective**  
This notebook compares SHAP and LIME explanations by analyzing their **standard deviation** across multiple runs or samples. It focuses on the stability and robustness of explanations.

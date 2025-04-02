# Apnea Detection

## Overview
This repository contains code for processing ECG signals for sleep apnea detection. The project focuses on:
- **Preprocessing** ECG signals from the [Apnea-ECG dataset](https://www.physionet.org/content/apnea-ecg/) on PhysioNet.
- **Extracting Heart Rate Variability (HRV) features** using NeuroKit2.
- **Training and tuning machine learning models**, including both traditional classifiers and deep learning models.
- **Performing model selection and hyperparameter tuning** using various classifiers.
- **Reducing features** based on model-based importance.
- **Evaluating final models** using multiple performance metrics.

## Purpose
The goal of this project is to develop a practical application for real-time ECG analysis and sleep apnea detection using both interpretable features and end-to-end deep learning approaches.

## Dataset
We use the **[Apnea-ECG dataset](https://www.physionet.org/content/apnea-ecg/)**, which contains ECG recordings for sleep apnea detection. Please refer to the official dataset documentation for details on accessing and using the data.

## Models

### 1. Gradient Boosting on HRV Features
Out of 65 computed HRV-based features, only 7 were needed to achieve near-optimal accuracy. These were selected via model-based feature importance.

**Top 7 Features:**
1. **Total Power**
2. **Short-term variance of contributions of accelerations**
3. **80th percentile of the RR intervals**
4. **Percentage of short segments**
5. **Percentage of absolute differences in successive RR intervals greater than 20 ms**
6. **Higuchi’s Fractal Dimension**
7. **Median absolute deviation of the RR intervals divided by the median of the RR intervals**

**Performance:**

| Class            | Precision | Recall | F1-score | Support |
|------------------|-----------|--------|----------|---------|
| Non-Apnea        | 0.83      | 0.89   | 0.86     | 10286   |
| Apnea            | 0.80      | 0.71   | 0.75     | 6272    |
| **Accuracy**     |           |        | **0.82** | 16558   |
| **Macro Avg**    | 0.82      | 0.80   | 0.81     | 16558   |
| **Weighted Avg** | 0.82      | 0.82   | 0.82     | 16558   |

Final model: **Gradient Boosting**  
Final test accuracy: **82%**  
AUC: **0.895**

### 2. Convolutional Neural Network (CNN) on Raw ECG

In addition to feature-based modeling, a custom 1D Convolutional Neural Network was trained directly on raw ECG windows to capture complex time-domain patterns without hand-crafted features.

**Key CNN Highlights:**
- Input: Raw ECG segments (preprocessed and normalized).
- Architecture: Several 1D convolutional layers with dropout and batch normalization.
- Training: Balanced with class weights and early stopping.
- Evaluation: Performance comparable to state-of-the-art models from academic literature.

**CNN Accuracy:** **~89%** on the test set  
This demonstrates that end-to-end learning on raw signals can outperform feature-engineered methods in certain settings.

## Web Application
The web application allows users to upload their ECG recordings and obtain predictions in real time.  
Visit the website here: **[ECG Based Apnea Detection](https://ecg-based-apnea-detection.streamlit.app/)**

Uploaded data is processed using the methodology described in the project, and results are presented in an easy-to-understand format. Predictions are currently made using the feature-based model for interpretability.

## Disclaimer
This project is for educational purposes only and is not intended for clinical or diagnostic use.

## Citation
> Bernardini, Andrea, Andrea Brunello, Gian Luigi Gigli, Angelo Montanari, and Nicola Saccomanno.  
> *AIOSA: An approach to the automatic identification of obstructive sleep apnea events based on deep learning*.  
> Artificial Intelligence in Medicine 118 (2021): 102133.  
> DOI: [10.1016/j.artmed.2021.102133](https://doi.org/10.1016/j.artmed.2021.102133)

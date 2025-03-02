# Apnea Detection

## Overview
This repository contains code for processing ECG signals for sleep apnea detection. The project focuses on:
- **Preprocessing** ECG signals from the [Apnea-ECG dataset](https://www.physionet.org/content/apnea-ecg/) on PhysioNet.
- **Extracting Heart Rate Variability (HRV) features** using NeuroKit2.
- **Performing model selection and hyperparameter tuning** using various classifiers.
- **Reducing features** based on model-based importance.
- **Evaluating the final model** with multiple performance metrics.

## Purpose
The goal of this project is to develop a practical application for real-time ECG analysis and sleep apnea detection.

## Dataset
We use the **[Apnea-ECG dataset](https://www.physionet.org/content/apnea-ecg/)**, which contains ECG recordings for sleep apnea detection. Please refer to the official dataset documentation for details on accessing and using the data.

## Web Application
The web application allows users to upload their ECG recordings and obtain predictions in real time.  
Visit the website here: **[ECG Based Apnea Detection](https://ecg-based-apnea-detection.streamlit.app/)**

Uploaded data is processed using the methodology described in the project, and results are presented in an easy-to-understand format.

## Results
Out of 65 computed features, only 7 were needed to achieve near-optimal accuracy. The 7 best features are:
1. **Total Power**
2. **Short-term variance of contributions of accelerations**
3. **80th percentile of the RR intervals**
4. **Percentage of short segments**
5. **Percentage of absolute differences in successive RR intervals greater than 20 ms**
6. **Higuchi’s Fractal Dimension**
7. **Median absolute deviation of the RR intervals divided by the median of the RR intervals**

The classification report is as follows:

| Class            | Precision | Recall | F1-score | Support |
|------------------|-----------|--------|----------|---------|
| Non-Apnea        | 0.83      | 0.89   | 0.86     | 10286   |
| Apnea            | 0.80      | 0.71   | 0.75     | 6272    |
| **Accuracy**     |           |        | **0.82** | 16558   |
| **Macro Avg**    | 0.82      | 0.80   | 0.81     | 16558   |
| **Weighted Avg** | 0.82      | 0.82   | 0.82     | 16558   |

The final model achieves an accuracy of 82% and an AUC of 0.895 using only 7 features.

## Disclaimer
This project is for educational purposes only and is not intended for clinical or diagnostic use.

## Citation
> Bernardini, Andrea, Andrea Brunello, Gian Luigi Gigli, Angelo Montanari, and Nicola Saccomanno.  
> *AIOSA: An approach to the automatic identification of obstructive sleep apnea events based on deep learning*.  
> Artificial Intelligence in Medicine 118 (2021): 102133.  
> DOI: [10.1016/j.artmed.2021.102133](https://doi.org/10.1016/j.artmed.2021.102133)

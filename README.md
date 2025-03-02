# Apnea Detection Web Application

## Overview
This repository contains code for processing ECG signals for apnea detection. The project focuses on:
- Preprocessing ECG signals (from the Apnea ECG dataset on PhysioNet)
- Extracting Heart Rate Variability (HRV) features using NeuroKit2
- Performing model selection and hyperparameter tuning using various classifiers
- Reducing features based on model-based importance
- Evaluating the final model with multiple performance metrics

## Purpose


To create a practical application for real-time ECG analysis.

## Dataset

We use the **[Apnea-ECG dataset](https://www.physionet.org/content/apnea-ecg/)**, which contains electrocardiogram (ECG) recordings for sleep apnea detection. For details on how to access and use this dataset, please refer to the official dataset documentation.


## Web Application

The web application allows users to upload their ECG recordings and obtain predictions in real time.
Visit the website here: **[ECG Based Apnea Detection](https://ecg-based-apnea-detection.streamlit.app/)**
The uploaded data is processed using the same methodology described in the paper, and the results are displayed in an easy-to-understand format.

## Results
Oout of 65 computed features, only 7 were needed to achieve near best accuracy possible (with our methods). The 7 best features are:
1) Total Power
2) Short-term variance of contributions of accelerations
3) The 80th percentile of the RR intervals
4) Percentage of short segments
5) The percentage of absolute differences in successive RR intervals greater than 20 ms
6) Higuchi’s Fractal Dimension
7) The median absolute deviation of the RR intervals divided by the median of the RR intervals

The classification report is as follows:
| Class           | Precision | Recall | F1-score | Support |
|-----------------|-----------|--------|----------|---------|
| 0               | 0.83      | 0.89   | 0.86     | 10286   |
| 1               | 0.80      | 0.71   | 0.75     | 6272    |
| **Accuracy**    |           |        | **0.82** | 16558   |
| **Macro Avg**   | 0.82      | 0.80   | 0.81     | 16558   |
| **Weighted Avg**| 0.82      | 0.82   | 0.82     | 16558   |
The model has accuracy of 82% with only 7 features and has AUC of 0.895. 


## Disclaimer

This project is for educational purposes only. It is not intended for clinical or diagnostic use.
## Citation
> Bernardini, Andrea, Andrea Brunello, Gian Luigi Gigli, Angelo Montanari, and Nicola Saccomanno.  
> *AIOSA: An approach to the automatic identification of obstructive sleep apnea events based on deep learning*.  
> Artificial Intelligence in Medicine 118 (2021): 102133.  
> DOI: [10.1016/j.artmed.2021.102133](https://doi.org/10.1016/j.artmed.2021.102133)

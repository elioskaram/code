# Heart Murmur Detection from Phonocardiogram Recordings

## Project Overview
This project focuses on developing a machine learning model to detect heart murmurs using phonocardiogram (PCG) recordings. The goal is to classify murmurs into three categories: `Absent`, `Present`, and `Unknown`, leveraging signal processing and supervised learning techniques.

## Objectives
- Automate the detection of heart murmurs to assist clinicians.
- Process audio data to extract meaningful features related to cardiac health.
- Build a robust classification model despite imbalanced datasets.

## Dataset
- **Sources**: Provided by the George B. Moody PhysioNet Challenge 2022.
- **Data Types**:
  - **Audio Files (.wav)**: Contain PCG recordings.
  - **Segmentation Files (.tsv)**: Indicate systolic and diastolic phases.
  - **Metadata Files (.txt)**: Include demographic and clinical information (age, sex, heart rate, etc.).
  
## Methodology
1. **Data Preprocessing**:
   - Noise reduction and normalization of audio signals.
   - Extraction of features like heart rate variability, signal statistics (mean, RMS, etc.).
2. **Feature Engineering**:
   - Derived features from PCG recordings and metadata.
3. **Model Development**:
   - Used Random Forest Classifier with hyperparameter tuning.
   - Addressed class imbalance with SMOTE oversampling.
4. **Evaluation**:
   - Metrics: Precision, Recall, F1-Score, Accuracy.
   - Confusion matrix for error analysis.

## Results
- Achieved **73% accuracy** on the test set.
- Strong performance in detecting `Absent` murmurs but challenges with `Present` and `Unknown` classes due to class imbalance.

## Tools and Libraries
- **Programming Language**: Python
- **Key Libraries**: NumPy, Pandas, Scikit-learn, imbalanced-learn (SMOTE), SciPy, Matplotlib.

## Challenges
- Class imbalance affected model generalization for minority classes.
- Difficulty in extracting audio features that reliably distinguish murmur categories.


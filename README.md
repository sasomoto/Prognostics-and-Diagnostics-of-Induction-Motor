# Prognostics-and-Diagnostics-of-Induction-Motor
A comprehensive repository for an advanced Induction Motor Diagnostics and Prognostics algorithm. This system leverages deep learning, signal processing, and fault analysis techniques to predict motor health, classify fault conditions, and estimate the Remaining Useful Life (RUL) of the motor.
Features
Fault Classification:

Automatically detect and classify motor conditions into states: Healthy, R0S25, R0S50, R0S75.
Prognostics:

Estimate the Remaining Useful Life (RUL) of the motor using scalable deep learning models.
Data Processing:

Efficiently preprocess .lvm (LabVIEW Measurement) files and convert them into CSV for analysis.
Combine and organize data from multiple runs for training and validation.
Memory-Efficient Training:

Train models with large datasets using a custom data generator to minimize memory usage.
Integrated support for GPU memory management.
Deep Learning Model Architecture:

Hybrid architecture combining 1D Convolutional Neural Networks (CNN) for feature extraction and LSTMs for sequence modeling.
Multi-task output branches for classification and RUL prediction.
Evaluation:

Comprehensive performance metrics:
Classification: Accuracy, Precision, Recall, F1-Score.
RUL Regression: MSE, RMSE, MAE, R².
Visualization of training history, confusion matrices, and RUL predictions.
Custom Callbacks:

Early stopping and best model checkpointing based on validation loss.
Technologies Used
Programming Languages: Python
Libraries and Frameworks:
TensorFlow/Keras for deep learning.
Pandas, NumPy for data manipulation.
Scikit-learn for metrics and preprocessing.
Matplotlib, Seaborn for visualization.
File Handling: .lvm to .csv converters for LabVIEW data.

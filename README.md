# 210 NLA Project

This project implements and compares three different machine learning models for breast cancer classification: Coordinate Descent, Stochastic Gradient Descent (SGD), and Support Vector Machine (SVM).

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

You can install the required packages using:

`pip install numpy pandas scikit-learn matplotlib seaborn`


## Dataset

The project uses the Breast Cancer Wisconsin (Diagnostic) Dataset, which is automatically downloaded from the UCI Machine Learning Repository.

## Usage

1. Clone the repository:

   `git clone https://github.com/imnishanth/nla-bcc.git`

   `cd nla-bcc`


2. Run the main script:

   `python main.py`


3. The script will output the following:
- Performance metrics (Accuracy, Precision, Recall, F1-score) for each model
- Learning curves for Coordinate Descent and SGD
- Confusion matrices for each model
- ROC curves for all models
- Best parameters for the SVM model

## Code Structure

- `main.py`: The main script that runs the entire pipeline
- `load_data()`: Loads and preprocesses the dataset
- `coordinate_descent_closed_form()`: Implements the Coordinate Descent algorithm
- `sgd()`: Implements the Stochastic Gradient Descent algorithm
- `evaluate_model()`: Calculates performance metrics for a given model
- `plot_confusion_matrix()`: Plots the confusion matrix
- `plot_roc_curve()`: Plots the ROC curve

## Experiment Results

Here are the results of our experiments:

1. Coordinate Descent:
- Accuracy: 0.9649
- Precision: 0.9545
- Recall: 0.9545
- F1-score: 0.9545

2. Stochastic Gradient Descent (SGD):
- Accuracy: 0.9561
- Precision: 0.9535
- Recall: 0.9318
- F1-score: 0.9425

3. Support Vector Machine (SVM):
- Accuracy: 0.9737
- Precision: 0.9787
- Recall: 0.9545
- F1-score: 0.9664

SVM Best Parameters:
- C: 1
- gamma: scale
- kernel: rbf

The SVM model performed the best among the three models, achieving the highest accuracy, precision, and F1-score. The Coordinate Descent and SGD models also performed well, with accuracies above 95%.

The learning curves, confusion matrices, and ROC curves provide additional insights into the models performance and can be found in the `main.ipynb` file.

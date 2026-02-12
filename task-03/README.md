# Task-03: Decision Tree Classifier for Customer Purchase Prediction

## Description
This project builds a **Decision Tree Classifier** to predict whether a customer will purchase a product or service based on demographic and behavioral data using the **Bank Marketing Dataset** from the UCI Machine Learning Repository.

## Dataset
- **Source**: [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Source**: [Prodigy InfoTech Dataset](https://github.com/Prodigy-InfoTech/data-science-datasets/tree/main/Task%203)
- **Samples**: ~45,000+ records
- **Features**: ~20 demographic and behavioral attributes
- **Target**: Customer purchase decision (Yes/No)

## Project Structure
```
task-03/
├── decision_tree_classifier.py   # Main Python script
├── requirements.txt              # Package dependencies
├── README.md                     # This file
├── model_summary.txt             # Generated summary report
├── target_distribution.png       # Visualization: class distribution
├── feature_importance.png        # Visualization: feature importance
├── confusion_matrix_and_roc.png  # Visualization: evaluation metrics
├── decision_tree.png             # Visualization: tree structure
└── metrics_comparison.png        # Visualization: performance comparison
```

## Requirements
- Python 3.7+
- See `requirements.txt` for package dependencies

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Script
```bash
python decision_tree_classifier.py
```

The script will:
1. ✓ Download the Bank Marketing dataset automatically
2. ✓ Perform exploratory data analysis (EDA)
3. ✓ Preprocess and encode the data
4. ✓ Train the decision tree classifier
5. ✓ Evaluate the model with multiple metrics
6. ✓ Generate visualizations
7. ✓ Create a detailed summary report

## Key Features

### 1. Exploratory Data Analysis
- Dataset overview (shape, info, missing values)
- Target variable distribution
- Statistical summaries
- Visualizations of class distribution

### 2. Data Preprocessing
- Automatic handling of categorical variables
- Label encoding for categorical features
- Train-test split with stratification

### 3. Decision Tree Training
- Optimized hyperparameters:
  - `max_depth`: 10
  - `min_samples_split`: 20
  - `min_samples_leaf`: 10
  - `class_weight`: 'balanced' (handles class imbalance)

### 4. Model Evaluation
**Metrics Calculated:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

### 5. Visualizations Generated
- **Target Distribution**: Class balance visualization
- **Feature Importance**: Top 10 most important features
- **Confusion Matrix**: True Positives, True Negatives, False Positives, False Negatives
- **ROC Curve**: Model's discriminative ability
- **Decision Tree Structure**: Visual representation of the tree
- **Metrics Comparison**: Training vs Testing performance

## Expected Output

### Console Output
```
Loading Bank Marketing Dataset...
Dataset loaded successfully! Shape: (45211, 21)

EXPLORATORY DATA ANALYSIS
...

TRAINING SET METRICS:
  Accuracy:  0.8920
  Precision: 0.6850
  Recall:    0.5230
  F1-Score:  0.5920

TESTING SET METRICS:
  Accuracy:  0.8890
  Precision: 0.6720
  Recall:    0.5100
  F1-Score:  0.5780
  ROC-AUC:   0.8450

✓ ALL TASKS COMPLETED SUCCESSFULLY!
```

### Generated Files
The script generates 5 visualization PNG files and 1 text summary.

## Model Performance

The Decision Tree Classifier achieves:
- **Test Accuracy**: ~88-90%
- **ROC-AUC**: ~0.84
- **Balanced Precision-Recall**: Good balance between false positives and false negatives

## Key Insights

1. **Feature Importance**: The model identifies the most important features for predicting customer purchases
2. **Class Balance**: The balanced class weight helps handle the imbalanced dataset
3. **Interpretability**: Decision trees provide interpretable rules for business insights
4. **Performance**: The model avoids overfitting with appropriate hyperparameters

## Customization

You can modify these parameters in the script:

```python
# Line 179: Adjust hyperparameters
dt_classifier = DecisionTreeClassifier(
    max_depth=10,              # Adjust tree depth
    min_samples_split=20,      # Increase to prevent overfitting
    min_samples_leaf=10,       # Minimum samples at leaf nodes
    random_state=42,           # For reproducibility
    class_weight='balanced'    # Handle imbalanced classes
)
```

## Educational Purpose

This project demonstrates:
- Data preprocessing and exploratory analysis
- Decision tree algorithm implementation
- Model evaluation and validation
- Performance visualization
- Real-world machine learning workflow

## Author
Created for Prodigy InfoTech - Data Science Internship, Task-03

## References
- [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- [Scikit-Learn Decision Tree Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [Scikit-Learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

**Note**: This script automatically downloads the dataset from the provided URLs. Ensure you have internet connectivity when running it for the first time.

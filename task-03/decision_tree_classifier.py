"""
Task-03: Decision Tree Classifier for Customer Purchase Prediction
Using Bank Marketing Dataset from UCI Machine Learning Repository
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# 1. LOAD THE DATASET
# ============================================================================
print("Loading Bank Marketing Dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv"

try:
    df = pd.read_csv(url, sep=';')
    print(f"Dataset loaded successfully! Shape: {df.shape}")
except Exception as e:
    print(f"Error loading from URL: {e}")
    print("Please download manually from:")
    print("https://github.com/Prodigy-InfoTech/data-science-datasets/tree/main/Task%203")
    exit()

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("EXPLORATORY DATA ANALYSIS")
print("="*70)

print("\nFirst few rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Variable Distribution:")
print(df['y'].value_counts())
print(f"\nTarget Variable Percentage:\n{df['y'].value_counts(normalize=True) * 100}")

# Visualization: Target Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Count plot
df['y'].value_counts().plot(kind='bar', ax=axes[0], color=['#FF6B6B', '#4ECDC4'])
axes[0].set_title('Target Variable Distribution (Count)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Purchase (y)')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['No', 'Yes'], rotation=0)

# Pie chart
df['y'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                             colors=['#FF6B6B', '#4ECDC4'])
axes[1].set_title('Target Variable Distribution (Percentage)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: target_distribution.png")
plt.close()

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================
print("\n" + "="*70)
print("DATA PREPROCESSING")
print("="*70)

# Create a copy for preprocessing
data = df.copy()

# Encode target variable
print("\nEncoding target variable...")
data['y'] = (data['y'] == 'yes').astype(int)

# Identify categorical and numerical columns
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('y')

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Encode categorical variables
print("\nEncoding categorical variables...")
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    le_dict[col] = le
    print(f"  {col}: {len(le.classes_)} classes")

print("\nPreprocessed Data Info:")
print(data.info())

# ============================================================================
# 4. FEATURE SELECTION & SPLIT
# ============================================================================
print("\n" + "="*70)
print("FEATURE ENGINEERING & DATA SPLITTING")
print("="*70)

X = data.drop('y', axis=1)
y = data['y']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Features correlation with target
print("\nTop 10 Features Correlated with Target Variable:")
correlations = pd.concat([X, y], axis=1).corr()['y'].sort_values(ascending=False)
print(correlations.head(11))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")
print(f"Training set - Target distribution:\n{y_train.value_counts()}")
print(f"Testing set - Target distribution:\n{y_test.value_counts()}")

# ============================================================================
# 5. TRAIN DECISION TREE CLASSIFIER
# ============================================================================
print("\n" + "="*70)
print("TRAINING DECISION TREE CLASSIFIER")
print("="*70)

# Train with optimal parameters
dt_classifier = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    class_weight='balanced'
)

dt_classifier.fit(X_train, y_train)
print("\n✓ Decision Tree Model trained successfully!")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Visualization: Feature Importance
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(10)
plt.barh(top_features['feature'], top_features['importance'], color='#4ECDC4')
plt.xlabel('Importance', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title('Top 10 Important Features in Decision Tree', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature_importance.png")
plt.close()

# ============================================================================
# 6. MODEL EVALUATION
# ============================================================================
print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

# Predictions
y_train_pred = dt_classifier.predict(X_train)
y_test_pred = dt_classifier.predict(X_test)
y_test_pred_proba = dt_classifier.predict_proba(X_test)[:, 1]

# Training Metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

print("\nTRAINING SET METRICS:")
print(f"  Accuracy:  {train_accuracy:.4f}")
print(f"  Precision: {train_precision:.4f}")
print(f"  Recall:    {train_recall:.4f}")
print(f"  F1-Score:  {train_f1:.4f}")

# Testing Metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)

print("\nTESTING SET METRICS:")
print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")
print(f"  ROC-AUC:   {test_roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=['No Purchase', 'Purchase']))

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# Confusion Matrix Heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Purchase', 'Purchase'],
            yticklabels=['No Purchase', 'Purchase'])
axes[0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
axes[1].plot(fpr, tpr, label=f'ROC Curve (AUC = {test_roc_auc:.4f})', 
             color='#4ECDC4', linewidth=2)
axes[1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('confusion_matrix_and_roc.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrix_and_roc.png")
plt.close()

# Decision Tree Visualization (simplified for large trees)
plt.figure(figsize=(25, 15))
plot_tree(dt_classifier, feature_names=X.columns, class_names=['No', 'Yes'],
          filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree Visualization', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('decision_tree.png', dpi=150, bbox_inches='tight')
print("✓ Saved: decision_tree.png")
plt.close()

# Metrics Comparison
metrics_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Training': [train_accuracy, train_precision, train_recall, train_f1],
    'Testing': [test_accuracy, test_precision, test_recall, test_f1]
}

metrics_df = pd.DataFrame(metrics_data)
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics_df))
width = 0.35

ax.bar(x - width/2, metrics_df['Training'], width, label='Training', color='#FF6B6B')
ax.bar(x + width/2, metrics_df['Testing'], width, label='Testing', color='#4ECDC4')

ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance: Training vs Testing', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_df['Metric'])
ax.legend()
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)

for i, v in enumerate(metrics_df['Training']):
    ax.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
for i, v in enumerate(metrics_df['Testing']):
    ax.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: metrics_comparison.png")
plt.close()

# ============================================================================
# 8. PREDICTIONS ON NEW DATA
# ============================================================================
print("\n" + "="*70)
print("MAKING PREDICTIONS ON NEW DATA")
print("="*70)

print("\nSample Predictions from Test Set (first 10 samples):")
results_df = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': y_test_pred[:10],
    'Probability (Purchase)': y_test_pred_proba[:10]
})
results_df['Actual'] = results_df['Actual'].map({0: 'No', 1: 'Yes'})
results_df['Predicted'] = results_df['Predicted'].map({0: 'No', 1: 'Yes'})
print(results_df)

# ============================================================================
# 9. SUMMARY REPORT
# ============================================================================
print("\n" + "="*70)
print("SUMMARY REPORT")
print("="*70)

summary = f"""
DECISION TREE CLASSIFIER - BANK MARKETING DATASET

Dataset Information:
  • Total Samples: {df.shape[0]}
  • Total Features: {df.shape[1] - 1}
  • Positive Class (Purchase): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.2f}%)
  • Negative Class (No Purchase): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.2f}%)

Model Configuration:
  • Algorithm: Decision Tree Classifier
  • Max Depth: 10
  • Min Samples Split: 20
  • Min Samples Leaf: 10
  • Class Weight: Balanced

Training Set Performance:
  • Accuracy:  {train_accuracy:.4f}
  • Precision: {train_precision:.4f}
  • Recall:    {train_recall:.4f}
  • F1-Score:  {train_f1:.4f}

Testing Set Performance:
  • Accuracy:  {test_accuracy:.4f}
  • Precision: {test_precision:.4f}
  • Recall:    {test_recall:.4f}
  • F1-Score:  {test_f1:.4f}
  • ROC-AUC:   {test_roc_auc:.4f}

Key Findings:
  • The model achieves {test_accuracy:.2%} accuracy on the test set
  • Top 3 Important Features: {', '.join(feature_importance.head(3)['feature'].tolist())}
  • The model shows balanced performance between precision and recall
  • ROC-AUC score of {test_roc_auc:.4f} indicates good discriminative ability

Generated Visualizations:
  ✓ target_distribution.png - Distribution of purchase and non-purchase classes
  ✓ feature_importance.png - Top 10 most important features
  ✓ confusion_matrix_and_roc.png - Confusion matrix and ROC curve
  ✓ decision_tree.png - Full decision tree structure
  ✓ metrics_comparison.png - Training vs testing metrics comparison
"""

print(summary)

# Save summary to file
with open('model_summary.txt', 'w') as f:
    f.write(summary)
print("✓ Saved: model_summary.txt")

print("\n" + "="*70)
print("✓ ALL TASKS COMPLETED SUCCESSFULLY!")
print("="*70)

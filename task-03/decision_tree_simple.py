"""
Task-03: Decision Tree Classifier for Customer Purchase Prediction
Using Bank Marketing Dataset (with synthetic fallback)
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

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# 1. LOAD OR GENERATE DATASET
# ============================================================================
print("Loading Bank Marketing Dataset...")

# Try multiple URLs
urls = [
    "https://raw.githubusercontent.com/Prodigy-InfoTech/data-science-datasets/main/Task%203/bank-additional-full.csv",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv",
]

df = None
for url in urls:
    try:
        df = pd.read_csv(url, sep=';')
        print(f"✓ Dataset loaded successfully from URL! Shape: {df.shape}")
        break
    except:
        continue

# Fallback: Generate synthetic dataset
if df is None:
    print("⚠ Could not download dataset. Generating synthetic data...")
    np.random.seed(42)
    n_samples = 2000
    
    df = pd.DataFrame({
        'age': np.random.randint(18, 95, n_samples),
        'job': np.random.choice(['admin', 'technician', 'services', 'management', 'retired', 'blue-collar', 'unemployed', 'entrepreneur'], n_samples),
        'marital': np.random.choice(['married', 'single', 'divorced'], n_samples),
        'education': np.random.choice(['primary', 'secondary', 'tertiary', 'unknown'], n_samples),
        'default': np.random.choice(['yes', 'no'], n_samples),
        'balance': np.random.randint(-1000, 100000, n_samples),
        'housing': np.random.choice(['yes', 'no'], n_samples),
        'loan': np.random.choice(['yes', 'no'], n_samples),
        'contact': np.random.choice(['telephone', 'cellular'], n_samples),
        'day': np.random.randint(1, 31, n_samples),
        'month': np.random.choice(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], n_samples),
        'duration': np.random.randint(0, 4000, n_samples),
        'campaign': np.random.randint(1, 50, n_samples),
        'pdays': np.random.randint(-1, 999, n_samples),
        'previous': np.random.randint(0, 10, n_samples),
        'poutcome': np.random.choice(['success', 'failure', 'unknown'], n_samples),
        'y': np.random.choice(['yes', 'no'], n_samples, p=[0.12, 0.88])
    })
    print(f"✓ Synthetic dataset generated! Shape: {df.shape}")

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("EXPLORATORY DATA ANALYSIS")
print("="*70)

print("\nDataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nTarget Variable Distribution:")
print(df['y'].value_counts())
print(f"\nTarget Percentage:\n{df['y'].value_counts(normalize=True) * 100}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['y'].value_counts().plot(kind='bar', ax=axes[0], color=['#FF6B6B', '#4ECDC4'])
axes[0].set_title('Target Distribution (Count)', fontweight='bold')
axes[0].set_xticklabels(['No', 'Yes'], rotation=0)
axes[0].set_ylabel('Count')

df['y'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4'])
axes[1].set_title('Target Distribution (%)', fontweight='bold')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: target_distribution.png")
plt.show()

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================
print("\n" + "="*70)
print("DATA PREPROCESSING")
print("="*70)

data = df.copy()
data['y'] = (data['y'] == 'yes').astype(int)

categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'y' in numerical_cols:
    numerical_cols.remove('y')

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Encode categorical variables
print("\nEncoding categorical variables...")
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    le_dict[col] = le

print(f"✓ Encoded {len(categorical_cols)} categorical columns")

# ============================================================================
# 4. FEATURE ENGINEERING & SPLIT
# ============================================================================
print("\n" + "="*70)
print("FEATURE ENGINEERING & DATA SPLITTING")
print("="*70)

X = data.drop('y', axis=1)
y = data['y']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Feature correlation with target
print("\nTop 10 Features Correlated with Target:")
correlations = pd.concat([X, y], axis=1).corr()['y'].sort_values(ascending=False)
print(correlations.head(11))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# ============================================================================
# 5. TRAIN DECISION TREE
# ============================================================================
print("\n" + "="*70)
print("TRAINING DECISION TREE CLASSIFIER")
print("="*70)

dt_classifier = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    class_weight='balanced'
)

dt_classifier.fit(X_train, y_train)
print("\n✓ Decision Tree trained successfully!")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Visualization
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(10)
plt.barh(top_features['feature'], top_features['importance'], color='#4ECDC4')
plt.xlabel('Importance', fontweight='bold')
plt.ylabel('Features', fontweight='bold')
plt.title('Top 10 Important Features', fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature_importance.png")
plt.show()

# ============================================================================
# 6. MODEL EVALUATION
# ============================================================================
print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

y_train_pred = dt_classifier.predict(X_train)
y_test_pred = dt_classifier.predict(X_test)
y_test_pred_proba = dt_classifier.predict_proba(X_test)[:, 1]

# Training metrics
train_acc = accuracy_score(y_train, y_train_pred)
train_prec = precision_score(y_train, y_train_pred, zero_division=0)
train_rec = recall_score(y_train, y_train_pred, zero_division=0)
train_f1 = f1_score(y_train, y_train_pred, zero_division=0)

print("\nTRAINING METRICS:")
print(f"  Accuracy:  {train_acc:.4f}")
print(f"  Precision: {train_prec:.4f}")
print(f"  Recall:    {train_rec:.4f}")
print(f"  F1-Score:  {train_f1:.4f}")

# Testing metrics
test_acc = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred, zero_division=0)
test_rec = recall_score(y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
test_roc = roc_auc_score(y_test, y_test_pred_proba)

print("\nTESTING METRICS:")
print(f"  Accuracy:  {test_acc:.4f}")
print(f"  Precision: {test_prec:.4f}")
print(f"  Recall:    {test_rec:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")
print(f"  ROC-AUC:   {test_roc:.4f}")

# Confusion Matrix & Report
cm = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['No Purchase', 'Purchase']))

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
axes[0].set_title('Confusion Matrix', fontweight='bold')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
axes[1].plot(fpr, tpr, label=f'ROC (AUC={test_roc:.4f})', color='#4ECDC4', linewidth=2)
axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('confusion_matrix_and_roc.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrix_and_roc.png")
plt.show()

# Metrics comparison
metrics_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Training': [train_acc, train_prec, train_rec, train_f1],
    'Testing': [test_acc, test_prec, test_rec, test_f1]
}

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics_data['Metric']))
width = 0.35

ax.bar(x - width/2, metrics_data['Training'], width, label='Training', color='#FF6B6B')
ax.bar(x + width/2, metrics_data['Testing'], width, label='Testing', color='#4ECDC4')

ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Model Performance: Training vs Testing', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_data['Metric'])
ax.legend()
ax.set_ylim([0, 1])

for i, (train, test) in enumerate(zip(metrics_data['Training'], metrics_data['Testing'])):
    ax.text(i - width/2, train + 0.02, f'{train:.3f}', ha='center', fontsize=9)
    ax.text(i + width/2, test + 0.02, f'{test:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: metrics_comparison.png")
plt.show()

# Decision Tree (simplified visualization for large trees)
plt.figure(figsize=(20, 12))
plot_tree(dt_classifier, feature_names=X.columns, class_names=['No', 'Yes'],
          filled=True, rounded=True, fontsize=8, max_depth=3)
plt.title('Decision Tree (Depth 3 visualization)', fontweight='bold', fontsize=16)
plt.tight_layout()
plt.savefig('decision_tree.png', dpi=150, bbox_inches='tight')
print("✓ Saved: decision_tree.png")
plt.show()

# ============================================================================
# 8. PREDICTIONS ON NEW DATA
# ============================================================================
print("\n" + "="*70)
print("SAMPLE PREDICTIONS (First 10 Test Samples)")
print("="*70)

results = pd.DataFrame({
    'Actual': ['Yes' if x == 1 else 'No' for x in y_test.values[:10]],
    'Predicted': ['Yes' if x == 1 else 'No' for x in y_test_pred[:10]],
    'Probability': np.round(y_test_pred_proba[:10], 3)
})
print(results)

# ============================================================================
# 9. SUMMARY
# ============================================================================
print("\n" + "="*70)
print("✓ TASK COMPLETED SUCCESSFULLY!")
print("="*70)

summary = f"""
DECISION TREE CLASSIFIER - BANK MARKETING PREDICTION

Dataset: {df.shape[0]} samples, {X.shape[1]} features
Target: {(y == 1).sum()} purchases, {(y == 0).sum()} non-purchases

MODEL PERFORMANCE:
  Training - Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}
  Testing  - Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}
  ROC-AUC: {test_roc:.4f}

VISUALIZATIONS GENERATED:
  ✓ target_distribution.png
  ✓ feature_importance.png
  ✓ confusion_matrix_and_roc.png
  ✓ decision_tree.png
  ✓ metrics_comparison.png
"""

print(summary)

with open('model_summary.txt', 'w') as f:
    f.write(summary)

print("✓ Saved: model_summary.txt")

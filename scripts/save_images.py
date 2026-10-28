import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
metrics_path = os.path.join(base_dir, 'artifacts', 'metrics.json')
feat_path = os.path.join(base_dir, 'artifacts', 'feature_importance.json')
data_path = os.path.join(base_dir, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
img_dir = os.path.join(base_dir, 'images')
os.makedirs(img_dir, exist_ok=True)

with open(metrics_path, 'r', encoding='utf-8') as f:
    metrics = json.load(f)
with open(feat_path, 'r', encoding='utf-8') as f:
    feats = json.load(f)
df = pd.read_csv(data_path)

champ = metrics['champion_model']
champ_metrics = metrics['test_evaluation'][champ]

# 1. Confusion Matrix
cm = np.array(champ_metrics['confusion_matrix'])
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted: No Churn', 'Predicted: Churn'],
            yticklabels=['Actual: No Churn', 'Actual: Churn'],
            annot_kws={'size': 14, 'weight': 'bold'})
plt.title(f'Held-Out Test Set Confusion Matrix - {champ} (N=1,409)', fontsize=13, weight='bold', pad=15)
plt.ylabel('True Class', fontsize=12)
plt.xlabel('Predicted Class', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'confusion_matrix.png'), dpi=200)
plt.close()

# 2. ROC & PR Curves
roc_data = champ_metrics['roc_curve']
pr_data = champ_metrics['pr_curve']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(roc_data['fpr'], roc_data['tpr'], color='#1a73e8', lw=2.5,
         label=f"{champ} (AUC = {champ_metrics['roc_auc']:.3f})")
ax1.plot([0, 1], [0, 1], color='grey', linestyle='--', lw=1.5, label='Random Chance')
ax1.set_title('Receiver Operating Characteristic (ROC)', fontsize=12, weight='bold')
ax1.set_xlabel('False Positive Rate', fontsize=11)
ax1.set_ylabel('True Positive Rate (Recall)', fontsize=11)
ax1.legend(loc='lower right')
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.plot(pr_data['recall'], pr_data['precision'], color='#34a853', lw=2.5,
         label=f"{champ} (F1 = {champ_metrics['f1_score']:.3f})")
ax2.axhline(y=0.2654, color='grey', linestyle='--', lw=1.5, label='Baseline Prevalence (26.5%)')
ax2.set_title('Precision-Recall Curve', fontsize=12, weight='bold')
ax2.set_xlabel('Recall', fontsize=11)
ax2.set_ylabel('Precision', fontsize=11)
ax2.legend(loc='upper right')
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'roc_pr_curves.png'), dpi=200)
plt.close()

# 3. Top Feature Importance
top_feats = pd.DataFrame(feats).head(12)
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=top_feats, palette='Blues_r')
plt.title(f'Top 12 Churn Predictive Features - {champ}', fontsize=13, weight='bold', pad=12)
plt.xlabel('Relative Feature Importance Score', fontsize=11)
plt.ylabel('Feature Name', fontsize=11)
plt.grid(axis='x', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'feature_importance.png'), dpi=200)
plt.close()

# 4. EDA 4-Panel Overview
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
sns.countplot(x='Contract', hue='Churn', data=df, palette=['#1a73e8', '#d93025'], ax=axs[0, 0])
axs[0, 0].set_title('Churn Count by Contract Type', fontsize=12, weight='bold')

sns.countplot(x='InternetService', hue='Churn', data=df, palette=['#1a73e8', '#d93025'], ax=axs[0, 1])
axs[0, 1].set_title('Churn Count by Internet Service', fontsize=12, weight='bold')

sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', palette=['#1a73e8', '#d93025'], bins=36, ax=axs[1, 0])
axs[1, 0].set_title('Customer Tenure Distribution (Months)', fontsize=12, weight='bold')

df['TotalChargesNum'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce').fillna(0)
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette=['#1a73e8', '#d93025'], ax=axs[1, 1])
axs[1, 1].set_title('Monthly Charges Boxplot by Churn', fontsize=12, weight='bold')

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'eda_churn_distributions.png'), dpi=200)
plt.close()

print('All 4 ML output images generated successfully in:', img_dir)


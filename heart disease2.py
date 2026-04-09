# ---- Code ----
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline

import os
print(os.listdir())

import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense


# ---- Code ----
df = pd.read_csv("heart.csv")


# ---- Code ----
df.head()


# ---- Code ----
df.shape


# ---- Code ----
df.info()


# ---- Code ----
df.describe()


# ---- Code ----
df.isnull().sum()


# ---- Code ----
plt.figure(figsize=(20,20))
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()


# ---- Code ----
df.duplicated().sum()


# ---- Code ----
df = df.drop_duplicates()


# ---- Code ----
print("Clean shape:", df.shape)

print(df['target'].value_counts())


# ---- Code ----
counts = df['target'].value_counts()

fig, ax = plt.subplots(1, 2, figsize=(10,4))

ax[0].bar(['No Disease', 'Disease'], counts, color=['lightblue','red'])
ax[0].set_title('Class Distribution')

ax[1].pie(counts, labels=['No Disease', 'Disease'], autopct='%1.1f%%' ,colors=['lightblue','red'])
ax[1].set_title('Class Ratio')
plt.suptitle('Target Balance')
plt.show()
print(counts)
print(f"Imbalance ratio: {counts[0]/counts[1]:.2f}")


# ---- Code ----
continuous=['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
print(f"Continuous: {continuous}")
print(f"Categorical: {categorical}")
print("Target: target")


# ---- Code ----
fig, ax = plt.subplots(2, len(continuous), figsize=(16,12))
for i, col in enumerate(continuous):
    
    sns.histplot(df, x=col, hue='target', kde=True, ax=ax[0,i])
    ax[0,i].set_title(col)
    
    sns.boxplot(x='target', y=col, data=df, ax=ax[1,i])
    ax[1,i].set_title(f'{col} by Target')


# ---- Code ----
mean_values_by_target =df.groupby('target')[continuous].mean().reset_index().T
mean_values_by_target


# ---- Code ----
median_values_by_target =df.groupby('target')[continuous].median().reset_index().T
median_values_by_target


# ---- Code ----
df.agg(['mean' , 'median']).T


# ---- Code ----
for col in continuous:
    IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
    outliers = df[(df[col] < df[col].quantile(0.25) - 1.5*IQR) |
                  (df[col] > df[col].quantile(0.75) + 1.5*IQR)]
    
    print(f"{col}: {len(outliers)}")


# ---- Code ----
fig, ax = plt.subplots(2, 4, figsize=(18,8))
ax = ax.flatten()

for i, col in enumerate(categorical):
    df.groupby([col, 'target']).size().unstack().plot(kind='bar', ax=ax[i] , color=['lightblue','red'])
    ax[i].set_title(col)


plt.tight_layout()
plt.show()


# ---- Code ----
print("=== Disease Rate % per Category ===\n")

for col in categorical:
    print(col)
    print(df.groupby(col)['target'].mean()*100)


# ---- Code ----
corr = df.corr()

plt.figure(figsize=(12,8))
sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
plt.title('Correlation Heatmap')
plt.show()

print("\nTop features correlated with target:\n")
print(corr['target'].drop('target').sort_values(ascending=False))


# ---- Code ----
top_features = ['thalach', 'oldpeak',"slope", 'age', 'cp', 'target']

sns.pairplot(df[top_features], hue='target',
             palette={0: '#00b4d8', 1: '#e63946'},
             diag_kind='kde', plot_kws={'alpha': 0.5})

plt.suptitle('Pairplot — Top Features vs Target', y=1.02, fontsize=13)
plt.show()


# ---- Code ----
from scipy.stats import ttest_ind

for col in continuous:
    group0 = df[df['target'] == 0][col]
    group1 = df[df['target'] == 1][col]
    
    stat, p = ttest_ind(group0, group1)
    print(f"{col}: t-stat={stat:.2f}, p-value={p:.4f}")


# ---- Code ----
from scipy.stats import chi2_contingency
print(f"{'Feature':<12} {'Chi2':>10} {'p-value':>12} {'Significant':>12}")
for col in categorical:
    contingency_table = pd.crosstab(df[col], df['target'])
    stat, p, dof, expected = chi2_contingency(contingency_table)
    significant = 'Yes' if p < 0.05 else 'No'
    print(f"{col:<12} {stat:>10.2f} {p:>12.4f} {significant:>12}")


# ---- Code ----
df_features=df.copy()


# ---- Code ----
# thalach → low max heart rate = high risk
df_features['low_thalach'] = (df_features['thalach'] < 140).astype(int)

# oldpeak → ST depression > 2 = ischemia
df_features['high_oldpeak'] = (df_features['oldpeak'] > 2).astype(int)

# age → risk increases after 55
df_features['age_group'] = pd.cut(df_features['age'],
                                   bins=[0, 40, 55, 70, 100],
                                   labels=[0, 1, 2, 3]).astype(int)

# age × thalach → older + low heart rate = danger
df_features['age_thalach'] = df_features['age'] * df_features['thalach']


# ---- Code ----
# ca > 0 → any blocked vessel = risk
df_features['has_blockage'] = (df_features['ca'] > 0).astype(int)

# exang + high_oldpeak → combined exercise stress score
df_features['exercise_risk'] = df_features['exang'] + df_features['high_oldpeak']

# ── Based on weak features 

# chol was weak → normalize by age to extract signal
df_features['chol_per_age'] = (df_features['chol'] / df_features['age']).round(2)

# fbs was weak → combine with chol to make metabolic risk
df_features['metabolic_risk'] = df_features['fbs'] + (df_features['chol'] > 200).astype(int)

print("New shape:", df_features.shape)
print("\nNew features added: 8")


# ---- Code ----
df.shape, df_features.shape


# ---- Code ----
df.columns


# ---- Code ----
new_features = ['low_thalach', 'high_oldpeak', 'age_group',
                'age_thalach', 'has_blockage', 'exercise_risk',
                'chol_per_age', 'metabolic_risk']
corr = df_features[new_features + ['target']].corr()['target'].drop('target')
corr_df = corr.abs().sort_values(ascending=False).round(3).reset_index()
corr_df.columns = ['Feature', 'Correlation with Target']
print(corr_df)


# ---- Code ----
df_features = df_features.drop(columns=['chol_per_age', 'metabolic_risk'])


# ---- Code ----
print("Final shape:", df_features.shape)


# ---- Code ----
corr_all=df_features.corr()['target'].drop('target').abs().sort_values(ascending=False).round(3).reset_index()
corr_all.columns = ['Feature', 'Correlation with Target']
print(corr_all)


# ---- Code ----
plt.figure(figsize=(10, 8))
corr_all = df_features.corr()['target'].drop('target')
corr_all.sort_values().plot(kind='barh')
plt.axvline(x=0.1, color='red', linestyle='--', label='Threshold 0.1')
plt.title('Feature Correlation with Target')
plt.legend()
plt.show()


# ---- Code ----
df_features = df_features.drop(columns=['chol', 'fbs', 'age_thalach'])

print("Final shape:", df_features.shape)


# ---- Code ----
X=df_features.drop(columns=['target'])
y=df_features['target']


# ---- Code ----
X.shape, y.shape


# ---- Code ----
X.columns


# ---- Code ----
y.value_counts()


# ---- Code ----
continuous_features = ['age', 'trestbps', 'thalach', 'oldpeak']
categorical_features = ['sex', 'cp', 'restecg', 'exang', 'slope', 'ca', 'thal',
                        'low_thalach', 'high_oldpeak', 'age_group', 'has_blockage', 'exercise_risk']


# ---- Code ----
print(f"Continuous: {continuous_features}")
print(f"Categorical: {categorical_features}")


# ---- Code ----
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X[continuous_features] = scaler.fit_transform(X[continuous_features])
X[continuous_features].head(2)


# ---- Code ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  
)


# ---- Code ----
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report,roc_auc_score, roc_curve 


# ---- Code ----
models={
    'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced',random_state=42),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42,n_estimators=100),
    'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
    'XGBoost': XGBClassifier(verbosity=0, eval_metric='logloss', random_state=42,)
}


# ---- Code ----
##  Train & Evaluate All Models
results = {}

for name, model in models.items():
    # Cross validation
    cv_scores = cross_val_score(model, X_train, y_train, 
                                 cv=5, scoring='accuracy')
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    cv   = cv_scores.mean()
    
    results[name] = {
        'model':   model,
        'y_pred':  y_pred,
        'y_prob':  y_prob,
        'accuracy': acc,
        'auc':      auc,
        'cv_score': cv
    }
    
    print(f"{'='*45}")
    print(f"  {name}")
    print(f"  CV Accuracy : {cv:.3f} ± {cv_scores.std():.3f}")
    print(f"  Test Accuracy: {acc:.3f}")
    print(f"  AUC Score   : {auc:.3f}")


# ---- Code ----
summary = pd.DataFrame({
    name: {
        'CV Accuracy':   f"{v['cv_score']:.3f}",
        'Test Accuracy': f"{v['accuracy']:.3f}",
        'AUC Score':     f"{v['auc']:.3f}"
    }
    for name, v in results.items()
}).T

summary = summary.sort_values('AUC Score', ascending=False)
print("\n=== Model Comparison ===\n")
print(summary)


# ---- Code ----
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (name, v) in enumerate(results.items()):
    cm = confusion_matrix(y_test, v['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
    axes[i].set_title(f'{name} | AUC: {v["auc"]:.3f}')

plt.suptitle('Confusion Matrices', fontsize=14)
plt.tight_layout()
plt.show()


# ---- Code ----
for name, v in results.items():
    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    print(classification_report(y_test, v['y_pred'],
                                target_names=['No Disease','Disease']))
    print()


# ---- Code ----
plt.figure(figsize=(10, 7))

for name, v in results.items():
    fpr, tpr, _ = roc_curve(y_test, v['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {v['auc']:.3f})", linewidth=2)

plt.plot([0,1], [0,1], 'k--', alpha=0.5, label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — All 6 Models', fontweight='bold', fontsize=14)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


# ---- Code ----
from sklearn.model_selection import GridSearchCV
def tune_model(model, params, X_train, y_train):
    grid = GridSearchCV(model, params, 
                        cv=5, 
                        scoring='roc_auc',
                        n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    print(f"Best AUC:    {grid.best_score_:.3f}")
    return grid.best_estimator_ 


# ---- Code ----
 #1. Logistic Regression
print("=== Logistic Regression ===")
lr_best = tune_model(
    LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    {'C': [0.01, 0.1, 1, 10, 100]},
    X_train, y_train)

# 2. KNN
print("\n=== KNN ===")
knn_best = tune_model(
    KNeighborsClassifier(),
    {'n_neighbors': [3, 5, 7, 9, 11, 15],
     'weights': ['uniform', 'distance']},
    X_train, y_train)

# 3. Decision Tree
print("\n=== Decision Tree ===")
dt_best = tune_model(
    DecisionTreeClassifier(class_weight='balanced', random_state=42),
    {'max_depth': [3, 5, 7, 10],
     'min_samples_split': [2, 5, 10]},
    X_train, y_train)

# 4. Random Forest
print("\n=== Random Forest ===")
rf_best = tune_model(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    {'n_estimators': [100, 200],
     'max_depth': [5, 10, None],
     'min_samples_split': [2, 5]},
    X_train, y_train)

# 5. XGBoost
print("\n=== XGBoost ===")
xgb_best = tune_model(
    XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
    {'n_estimators': [100, 200],
     'max_depth': [3, 5, 7],
     'learning_rate': [0.01, 0.1, 0.3]},
    X_train, y_train)

# 6. SVM
print("\n=== SVM ===")
svm_best = tune_model(
    SVC(class_weight='balanced', random_state=42, probability=True),
    {'C': [0.1, 1, 10],
     'kernel': ['rbf', 'linear'],
     'gamma': ['scale', 'auto']},
    X_train, y_train)


# ---- Code ----
tuned_models = {
    'Logistic Regression': lr_best,
    'KNN':                 knn_best,
    'Decision Tree':       dt_best,
    'Random Forest':       rf_best,
    'XGBoost':             xgb_best,
    'SVM':                 svm_best
}

tuned_results = {}

for name, model in tuned_models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    tuned_results[name] = {
        'model':    model,
        'y_pred':   y_pred,
        'y_prob':   y_prob,
        'accuracy': accuracy_score(y_test, y_pred),
        'auc':      roc_auc_score(y_test, y_prob)
    }

# Compare before vs after
print(f"\n{'='*55}")
print(f"{'Model':<22} {'Before AUC':>12} {'After AUC':>12} {'Δ':>6}")
print(f"{'='*55}")
for name in results:
    before = results[name]['auc']
    after  = tuned_results[name]['auc']
    delta  = after - before
    symbol = '↑' if delta > 0 else '↓'
    print(f"{name:<22} {before:>12.3f} {after:>12.3f} {symbol}{abs(delta):.3f}")


# ---- Code ----
# ROC Curve - Tuned
plt.figure(figsize=(10, 7))

for name, v in tuned_results.items():
    fpr, tpr, _ = roc_curve(y_test, v['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {v['auc']:.3f})")

plt.plot([0,1], [0,1], 'k--')
plt.title('ROC Curve — Tuned Models')
plt.legend()
plt.show()


# ---- Code ----
# Confusion Matrix - Tuned
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (name, v) in enumerate(tuned_results.items()):
    cm = confusion_matrix(y_test, v['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
    axes[i].set_title(f'{name} | AUC: {v["auc"]:.3f}')

plt.suptitle('Confusion Matrices — Tuned Models', fontsize=14)
plt.tight_layout()
plt.show()


# ---- Code ----
import shap
# Use best model = Logistic Regression
best_model = tuned_results['Logistic Regression']['model']

# SHAP explainer
explainer = shap.LinearExplainer(best_model, X_train)
shap_values = explainer(X_test)


# ---- Code ----
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type='bar',
                  title='SHAP — Global Feature Importance')


# ---- Code ----
### feature importance for Random Forest
rf_model = tuned_results['Random Forest']['model']

importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

sns.barplot(data=importance_df, x='Importance', y='Feature')
plt.title('Feature Importance')
plt.show()


# ---- Code ----
### feature importance for Random Forest
rf_model = tuned_results['Decision Tree']['model']

importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

sns.barplot(data=importance_df, x='Importance', y='Feature')
plt.title('Feature Importance')
plt.show()


# ---- Markdown ----
# Clinical Patterns Discovered¶
# 1. Chest Pain Type (cp) is the #1 signal
# Patients with cp = 0 (asymptomatic) paradoxically showed the highest disease rate — silent heart disease is dangerous
# 
# 2. Vessel Blockage is critical
# has_blockage (ca > 0) became the #1 SHAP feature Any blocked vessel dramatically increases disease probability
# 
# 3. Max Heart Rate matters more than Cholesterol
# thalach (AUC correlation = 0.42) vs chol (0.08) Low max heart rate = poor cardiac response = high risk Cholesterol alone is not a reliable predictor here
# 
# 4. Exercise stress is a combined signal
# exercise_risk (exang + high_oldpeak) ranked #3 in importance Patients who experience angina AND ST depression during exercise are at significantly higher risk
# 
# 5. Age pattern is non-linear
# Risk increases sharply after age 55 But younger patients (40-55) with bad cp or ca are equally at risk → age alone is misleading


# ---- Markdown ----
# Model Verdict¶
# Model	AUC	Verdict
# Logistic Regression	0.885	Best overall
# SVM	0.885	Tied best
# Random Forest	0.870	Best ensemble
# XGBoost	0.839	Underperformed
# KNN	0.861	Improved after tuning
# Decision Tree	0.846	Most improved (+0.104)
# Logistic Regression outperforming XGBoost confirms that on small medical datasets, simpler models generalize better.
# 
# 


# ---- Markdown ----
# Limitations¶
# 
# Small dataset (302 patients) → results may not generalize
# Dataset is from 1988 Cleveland clinic → may be outdated
# XGBoost underperformed → needs more data to shine
# Model should assist doctors, not replace them


# ---- Markdown ----
# Final Takeaway
# 
# "The best predictor of heart disease in this dataset is not cholesterol or age — it is chest pain type, vessel blockage, and how the heart responds under exercise stress. These three signals together define cardiac risk."


# ---- Code ----

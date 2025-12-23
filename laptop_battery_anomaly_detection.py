import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
import shap

# Upload file in Colab
uploaded = files.upload()
filename = list(uploaded.keys())[0]
print(f"File uploaded: {filename}")

# Read the CSV file
df = pd.read_csv(filename)
print("File loaded successfully!")
print(df.head())

# Feature Engineering
df['degradation_rate'] = df['battery_health_percent'] / df['battery_age_months']
df['total_usage_hours'] = df['daily_usage_hours'] * df['battery_age_months']

# Select features for analysis
features = df[['battery_health_percent', 'battery_age_months', 'charging_cycles', 'daily_usage_hours', 'degradation_rate', 'total_usage_hours']]

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(features)

# Apply Isolation Forest for anomaly detection
iso_forest = IsolationForest(contamination=0.1)
anomalies_iso = iso_forest.fit_predict(X)

# Apply One-Class SVM
svm_model = OneClassSVM(nu=0.1)
anomalies_svm = svm_model.fit_predict(X)

# Apply DBSCAN
dbscan_model = DBSCAN(eps=0.5, min_samples=2)
anomalies_dbscan = dbscan_model.fit_predict(X)

# Add anomaly labels to the dataset
df['anomaly_iso'] = anomalies_iso
df['anomaly_svm'] = anomalies_svm
df['anomaly_dbscan'] = anomalies_dbscan

# Show all batteries
print("All batteries with anomaly labels:")
print(df)

# Scatter plot with clear legend
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
scatter = plt.scatter(X[:, 0], X[:, 1], c=anomalies_iso, cmap='viridis')
plt.colorbar(scatter, label='Anomaly (1: Normal, -1: Anomaly)')
plt.xlabel('Normalized Battery Health')
plt.ylabel('Normalized Battery Age')
plt.title('Isolation Forest')

plt.subplot(1, 3, 2)
scatter = plt.scatter(X[:, 0], X[:, 1], c=anomalies_svm, cmap='viridis')
plt.colorbar(scatter, label='Anomaly (1: Normal, -1: Anomaly)')
plt.xlabel('Normalized Battery Health')
plt.ylabel('Normalized Battery Age')
plt.title('One-Class SVM')

plt.subplot(1, 3, 3)
scatter = plt.scatter(X[:, 0], X[:, 1], c=anomalies_dbscan, cmap='viridis')
plt.colorbar(scatter, label='Anomaly (1: Normal, -1: Anomaly)')
plt.xlabel('Normalized Battery Health')
plt.ylabel('Normalized Battery Age')
plt.title('DBSCAN')

plt.tight_layout()
plt.show()

# Machine Learning Evaluation
print("\nMachine Learning Evaluation:")

# Isolation Forest
print("Isolation Forest Classification Report:")
print(classification_report(df['anomaly_iso'], anomalies_iso, target_names=['Anomaly', 'Normal']))
print("\nIsolation Forest Confusion Matrix:")
print(confusion_matrix(df['anomaly_iso'], anomalies_iso))

# One-Class SVM
print("\nOne-Class SVM Classification Report:")
print(classification_report(df['anomaly_svm'], anomalies_svm, target_names=['Anomaly', 'Normal']))
print("\nOne-Class SVM Confusion Matrix:")
print(confusion_matrix(df['anomaly_svm'], anomalies_svm))

# DBSCAN (mapped to two classes)
df['anomaly_dbscan_mapped'] = df['anomaly_dbscan'].apply(lambda x: -1 if x == -1 else 1)
anomalies_dbscan_mapped = df['anomaly_dbscan_mapped'].values
print("\nDBSCAN Classification Report:")
print(classification_report(df['anomaly_dbscan_mapped'], anomalies_dbscan_mapped, target_names=['Anomaly', 'Normal']))
print("\nDBSCAN Confusion Matrix:")
print(confusion_matrix(df['anomaly_dbscan_mapped'], anomalies_dbscan_mapped))

# ROC Curve for Isolation Forest
fpr, tpr, _ = roc_curve(df['anomaly_iso'], iso_forest.decision_function(X))
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Isolation Forest)')
plt.legend(loc="lower right")
plt.show()

# SHAP Values for Interpretability
explainer = shap.TreeExplainer(iso_forest)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, features, feature_names=features.columns.tolist())

# Predictive Maintenance Recommendations
print("\nPredictive Maintenance Recommendations:")
for index, row in df.iterrows():
    battery_id = row['device_id']
    health = row['battery_health_percent']
    age = row['battery_age_months']
    anomaly_iso = row['anomaly_iso']
    suggestion = ""
    
    if anomaly_iso == -1:
        suggestion = "This battery shows unusual degradation. Consider replacing soon."
    else:
        suggestion = f"This battery is healthy. Estimated remaining life: {max(60 - age, 0)} months (assuming 60-month lifespan)."
    
    print(f"{battery_id}: Health={health}%, Age={age} months, {suggestion}")

# Additional Visualizations
sns.pairplot(df[['battery_health_percent', 'battery_age_months', 'charging_cycles', 'daily_usage_hours', 'anomaly_iso']], hue='anomaly_iso')
plt.suptitle('Pairplot of Battery Features', y=1.02)
plt.show()

# Cross-Validation
cv_scores_iso = cross_val_score(iso_forest, X, anomalies_iso, cv=5, scoring='f1')
print("\nIsolation Forest Cross-Validation F1 Score:", cv_scores_iso.mean())

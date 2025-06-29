# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split 
# from sklearn.preprocessing import LabelEncoder
# from imblearn.over_sampling import RandomOverSampler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib

# # File paths
# data_path = r'C:\Users\PMYLS\Desktop\courses\pc\AI_Judicial_Decision_Support\AI_Judicial_Decision_Support\data\Pakistan_Crime_Dataset.csv'
# balanced_data_path = r'C:\Users\PMYLS\Desktop\courses\pc\AI_Judicial_Decision_Support\AI_Judicial_Decision_Support\data\Pakistan_Crime_Dataset_Balanced.csv'
# model_dir = r'C:\Users\PMYLS\Desktop\courses\pc\AI_Judicial_Decision_Support\AI_Judicial_Decision_Support\models'
# os.makedirs(model_dir, exist_ok=True)  # Create folder if it doesn't exist

# # Read dataset
# df = pd.read_csv(data_path)

# # Optional: Use smaller sample for faster testing
# df_sample = df.sample(frac=0.1, random_state=42)  # Use 10% of data
# df = df_sample  # comment this out if you want full data

# # Encode categorical columns
# label_encoders = {}
# for col in df.select_dtypes(include=['object']).columns:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
#     label_encoders[col] = le

# # Separate features and target
# X = df.drop(columns=['Sentencing Outcome'])
# y = df['Sentencing Outcome']

# # Balance dataset using RandomOverSampler instead of SMOTE
# ros = RandomOverSampler(random_state=42)
# X_resampled, y_resampled = ros.fit_resample(X, y)

# # Save balanced dataset to CSV
# df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
# df_balanced['Sentencing Outcome'] = y_resampled
# df_balanced.to_csv(balanced_data_path, index=False)
# print(f"Balanced dataset saved to {balanced_data_path}")

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
# )

# # Train RandomForest model with parallel processing and fewer trees for speed
# model = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
# model.fit(X_train, y_train)

# # Predict & evaluate
# y_pred = model.predict(X_test)
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# print("Confusion Matrix:")
# plt.figure(figsize=(10, 7))
# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
# plt.show()

# # Feature importance plot
# importances = pd.Series(model.feature_importances_, index=X.columns)
# importances.sort_values(ascending=False).plot(kind='bar', figsize=(10, 6))
# plt.title('Feature Importance')
# plt.show()

# # Save the model and label encoders
# model_path = os.path.join(model_dir, 'random_forest_model.pkl')
# joblib.dump(model, model_path)
# print(f"Model saved to {model_path}")

# encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
# joblib.dump(label_encoders, encoders_path)
# print(f"Label encoders saved to {encoders_path}")

# print("Model directory:", model_dir)
# print("Model path:", model_path)
# print("Encoders path:", encoders_path)


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# File paths
data_path = r'C:\Users\PMYLS\Desktop\courses\pc\AI_Judicial_Decision_Support\AI_Judicial_Decision_Support\data\Pakistan_Crime_Dataset.csv'
balanced_data_path = r'C:\Users\PMYLS\Desktop\courses\pc\AI_Judicial_Decision_Support\AI_Judicial_Decision_Support\data\Pakistan_Crime_Dataset_Balanced.csv'
model_dir = r'C:\Users\PMYLS\Desktop\courses\pc\AI_Judicial_Decision_Support\AI_Judicial_Decision_Support\models'
os.makedirs(model_dir, exist_ok=True)

# Read dataset
df = pd.read_csv(data_path)

# Optional: Use smaller sample for faster testing
df_sample = df.sample(frac=0.1, random_state=42)  # Use 10% of data
df = df_sample  # comment this out if you want full data

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate features and target
X = df.drop(columns=['Sentencing Outcome'])
y = df['Sentencing Outcome']

# Balance dataset using RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Save balanced dataset to CSV
df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
df_balanced['Sentencing Outcome'] = y_resampled
df_balanced.to_csv(balanced_data_path, index=False)
print(f"✅ Balanced dataset saved to: {balanced_data_path}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix Plot
print("📊 Confusion Matrix:")
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', linewidths=0.5, cbar=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Optionally save confusion matrix image
# plt.savefig("confusion_matrix.png")  # Uncomment to save image

# Feature importance plot
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).plot(kind='bar', figsize=(10, 6))
plt.title('🔍 Feature Importance')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.show()

# Save the model and label encoders
model_path = os.path.join(model_dir, 'random_forest_model.pkl')
joblib.dump(model, model_path)
print(f"✅ Model saved to: {model_path}")

encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
joblib.dump(label_encoders, encoders_path)
print(f"✅ Label encoders saved to: {encoders_path}")

print("📁 Model directory:", model_dir)

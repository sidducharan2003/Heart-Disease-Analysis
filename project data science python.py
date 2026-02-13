import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define column names and read the dataset
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target"
]
df = pd.read_csv("processed.cleveland.data", names=columns)

# Handle missing values and convert data types
df.replace("?", pd.NA, inplace=True)
df = df.dropna()
df = df.astype(float)

print(df.isnull().sum())
print(df.shape)
print(df.head())

# Display summary statistics of the dataset
print(df.describe())

# Plot frequency of each heart disease level
sns.countplot(x='target', data=df, hue='target', palette='Set2', legend=False)
plt.title("Heart Disease Frequency")
plt.xlabel("Heart Disease Level (0 = No Disease)")
plt.ylabel("Count")
plt.show()

# Plot age distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['age'], bins=20, kde=True, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Boxplot to compare age with presence of heart disease
plt.figure(figsize=(8, 5))
sns.boxplot(x='target', y='age', hue='target', data=df, palette='pastel', legend=False)
plt.title('Age vs Heart Disease')
plt.xlabel('Heart Disease (0 = No, 1+ = Yes)')
plt.ylabel('Age')
plt.show()

# Plot correlation heatmap
plt.figure(figsize=(12, 8))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Identify top 5 features most correlated with target
correlation_target = correlation['target'].abs().sort_values(ascending=False)
top_features = correlation_target[1:6].index.tolist()
print("Top 5 features most correlated with target:\n", top_features)

# Prepare data for model training
X = df[top_features]
y = (df['target'] > 0).astype(int)  # Convert target to binary (0 = no disease, 1 = disease)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# Train a logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Plot the ROC curve
from sklearn.metrics import roc_curve, roc_auc_score
y_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_probs):.2f}', color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid()
plt.show()

# Save the trained model for later use
import joblib
joblib.dump(model, 'heart_disease_model.pkl')

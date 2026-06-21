# iris-flower-classification
Successfully completed the Iris Flower Classification project using Python and Machine Learning. Built a classification model to predict Iris species with high accuracy and evaluated performance using a confusion matrix and classification metrics.

# Iris Flower Classification using Machine Learning

This project classifies Iris flowers into Setosa, Versicolor, and Virginica using machine learning algorithms such as Logistic Regression, KNN, and Decision Tree.

## Accuracy
100%

## Technologies Used
- Python
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

#program 
# Iris Flower Classification using Logistic Regression

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
iris = load_iris()

# Features and Target
X = iris.data
y = iris.target

# Create DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y

# Display First 5 Rows
print("Dataset Preview:")
print(df.head())

# Visualize Data
sns.pairplot(df, hue='species')
plt.show()

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='rocket')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Iris Classification Confusion Matrix")
plt.show()

# Predict New Flower
sample = [[5.1, 3.5, 1.4, 0.2]]
sample = scaler.transform(sample)

prediction = model.predict(sample)

species_names = iris.target_names
print("\nPredicted Species:", species_names[prediction[0]])

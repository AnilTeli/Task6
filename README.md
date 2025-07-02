# Task6
Iris Dataset KNN Project (Full Workflow)

Step 1: Data Loading

import numpy as np import pandas as pd import matplotlib.pyplot as plt import seaborn as sns from sklearn.datasets import load_iris

iris = load_iris() df = pd.DataFrame(data=iris.data, columns=iris.feature_names) df['target'] = iris.target

Step 2: Data Cleaning

print(df.isnull().sum()) print(df.describe())

Step 3: Preprocessing

from sklearn.preprocessing import StandardScaler

X = df.drop('target', axis=1) y = df['target']

scaler = StandardScaler() X_scaled = scaler.fit_transform(X)

Step 4: Outlier Detection

plt.figure(figsize=(12, 6)) sns.boxplot(data=pd.DataFrame(X_scaled, columns=iris.feature_names)) plt.title('Outlier Detection using Boxplot') plt.show()

Step 5: Feature Selection (Correlation Matrix)

plt.figure(figsize=(8, 6)) sns.heatmap(df.corr(), annot=True, cmap='coolwarm') plt.title('Feature Correlation Matrix') plt.show()

Step 6: Exploratory Data Analysis (EDA)

sns.pairplot(df, hue='target', palette='bright') plt.show()

sns.countplot(x='target', data=df) plt.title('Target Class Distribution') plt.show()

Step 7: Model Training with KNN

from sklearn.model_selection import train_test_split from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5) knn.fit(X_train, y_train)

Step 8: KNN Visualization (Decision Boundary)

from matplotlib.colors import ListedColormap

X_vis = X_scaled[:, :2] X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(X_vis, y, test_size=0.3, random_state=42)

knn_vis = KNeighborsClassifier(n_neighbors=5) knn_vis.fit(X_train_vis, y_train_vis)

x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1 y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1 xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()]) Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6)) plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(('red', 'green', 'blue'))) plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, edgecolor='k', cmap=ListedColormap(('red', 'green', 'blue'))) plt.title('KNN Decision Boundary (First Two Features)') plt.xlabel('Feature 1 (Scaled)') plt.ylabel('Feature 2 (Scaled)') plt.show()

Step 9: Model Evaluation

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = knn.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}") print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred)) print("Classification Report:\n", classification_report(y_test, y_pred))

Optional: Hyperparameter Tuning

k_values = range(1, 21) accuracies = []

for k in k_values: knn_k = KNeighborsClassifier(n_neighbors=k) knn_k.fit(X_train, y_train) y_pred_k = knn_k.predict(X_test) accuracies.append(accuracy_score(y_test, y_pred_k))

plt.figure(figsize=(10, 6)) plt.plot(k_values, accuracies, marker='o') plt.title('Accuracy vs. K Value') plt.xlabel('K') plt.ylabel('Accuracy') plt.xticks(k_values) plt.grid(True) plt.show()

best_k = k_values[np.argmax(accuracies)] print(f'Best K Value: {best_k} with Accuracy: {max(accuracies):.4f}')



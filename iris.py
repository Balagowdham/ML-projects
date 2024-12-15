import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the Iris dataset from a CSV file
file_path = 'D:\pythonproject\iris.data'  # Replace with your CSV file path
df = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print("Dataset Sample:")
print(df.head())

# Check if 'species' column exists in the DataFrame
if 'species' in df.columns:
    # Split the data into features and target
    X = df.drop('species', axis=1)  # Features: sepal_length, sepal_width, petal_length, petal_width
    y = df['species']  # Target variable: species

    # Convert categorical target variable to numerical labels (if needed)
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features (optional but recommended for KNN)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the K-Nearest Neighbors classifier
    knn = KNeighborsClassifier(n_neighbors=5)

    # Train the model
    knn.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = knn.predict(X_test)

    # Inverse transform numerical labels back to original categorical labels
    y_pred = encoder.inverse_transform(y_pred)
    y_test = encoder.inverse_transform(y_test)

    # Evaluate the model
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
else:
    print("Error: 'species' column not found in the dataset.")

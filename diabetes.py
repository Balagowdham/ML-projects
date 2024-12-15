import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the data
data = pd.read_csv(r'D:\pythonproject\diabetes.csv')

# Step 2: Preprocess the data
# Handle missing values, if any
data.fillna(data.mean(), inplace=True)

# Separate the features and the target
X = data.drop(columns='Outcome')
y = data['Outcome']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Step 5: Predict for new data
# Example new patient data (replace this with your actual data)
new_patient_data = pd.DataFrame({
    'Pregnancies': [5],
    'Glucose': [116],
    'BloodPressure': [74],
    'SkinThickness': [0],
    'Insulin': [0],
    'BMI': [25.6],
    'DiabetesPedigreeFunction': [0.201],
    'Age': [30]
})

# Standardize the new patient data
new_patient_data_scaled = scaler.transform(new_patient_data)

# Predict the outcome
new_prediction = model.predict(new_patient_data_scaled)
print(f'Predicted Outcome: {new_prediction[0]}')

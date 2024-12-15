import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load training data
train_data = pd.read_csv(r'D:\pythonproject\train.csv')

# Define the target and features
y = train_data['SalePrice']
X = train_data.drop(['SalePrice'], axis=1)

# Handle missing values and categorical variables
X['LotFrontage'] = X['LotFrontage'].fillna(X['LotFrontage'].median())
X['Alley'] = X['Alley'].fillna('NoAlley')
X['MasVnrType'] = X['MasVnrType'].fillna('None')
X['MasVnrArea'] = X['MasVnrArea'].fillna(0)
X['BsmtQual'] = X['BsmtQual'].fillna('NoBsmt')
X['BsmtCond'] = X['BsmtCond'].fillna('NoBsmt')
X['BsmtExposure'] = X['BsmtExposure'].fillna('NoBsmt')
X['BsmtFinType1'] = X['BsmtFinType1'].fillna('NoBsmt')
X['BsmtFinType2'] = X['BsmtFinType2'].fillna('NoBsmt')
X['Electrical'] = X['Electrical'].fillna(X['Electrical'].mode()[0])
X['FireplaceQu'] = X['FireplaceQu'].fillna('NoFireplace')
X['GarageType'] = X['GarageType'].fillna('NoGarage')
X['GarageFinish'] = X['GarageFinish'].fillna('NoGarage')
X['GarageQual'] = X['GarageQual'].fillna('NoGarage')
X['GarageCond'] = X['GarageCond'].fillna('NoGarage')
X['PoolQC'] = X['PoolQC'].fillna('NoPool')
X['Fence'] = X['Fence'].fillna('NoFence')
X['MiscFeature'] = X['MiscFeature'].fillna('None')

# Convert categorical variables to numerical values
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Validate the model
val_predictions = model.predict(X_val)
val_mae = mean_absolute_error(y_val, val_predictions)
print(f"Validation MAE: {val_mae}")

# Load test data
test_data = pd.read_csv(r'D:\pythonproject\test.csv')

# Handle missing values and categorical variables in test data
test_data['LotFrontage'] = test_data['LotFrontage'].fillna(test_data['LotFrontage'].median())
test_data['Alley'] = test_data['Alley'].fillna('NoAlley')
test_data['MasVnrType'] = test_data['MasVnrType'].fillna('None')
test_data['MasVnrArea'] = test_data['MasVnrArea'].fillna(0)
test_data['BsmtQual'] = test_data['BsmtQual'].fillna('NoBsmt')
test_data['BsmtCond'] = test_data['BsmtCond'].fillna('NoBsmt')
test_data['BsmtExposure'] = test_data['BsmtExposure'].fillna('NoBsmt')
test_data['BsmtFinType1'] = test_data['BsmtFinType1'].fillna('NoBsmt')
test_data['BsmtFinType2'] = test_data['BsmtFinType2'].fillna('NoBsmt')
test_data['Electrical'] = test_data['Electrical'].fillna(test_data['Electrical'].mode()[0])
test_data['FireplaceQu'] = test_data['FireplaceQu'].fillna('NoFireplace')
test_data['GarageType'] = test_data['GarageType'].fillna('NoGarage')
test_data['GarageFinish'] = test_data['GarageFinish'].fillna('NoGarage')
test_data['GarageQual'] = test_data['GarageQual'].fillna('NoGarage')
test_data['GarageCond'] = test_data['GarageCond'].fillna('NoGarage')
test_data['PoolQC'] = test_data['PoolQC'].fillna('NoPool')
test_data['Fence'] = test_data['Fence'].fillna('NoFence')
test_data['MiscFeature'] = test_data['MiscFeature'].fillna('None')

# Convert categorical variables to numerical values in test data
for column in test_data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    test_data[column] = le.fit_transform(test_data[column])

# Make predictions on the test set
test_predictions = model.predict(test_data)

# Save predictions to a CSV file
output = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': test_predictions})
output.to_csv('D:\pythonproject\submission.csv', index=False)

print("Model training and testing completed. Predictions saved to submission.csv.")

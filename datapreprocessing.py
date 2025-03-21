import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_data():
    # Load data
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    
    # Handle missing values
    train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
    test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
    test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)
    
    # Feature selection
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X_train = train_data[features]
    y_train = train_data['Survived']
    X_test = test_data[features]
    
    # Convert 'Sex' to numerical
    X_train['Sex'] = X_train['Sex'].map({'female': 0, 'male': 1})
    X_test['Sex'] = X_test['Sex'].map({'female': 0, 'male': 1})
    
    # Normalize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, 'scaler.joblib')
    
    return X_train_scaled, y_train, X_test_scaled

if _name_ == "_main_":
    preprocess_data()

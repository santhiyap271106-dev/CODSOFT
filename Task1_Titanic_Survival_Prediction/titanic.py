import pandas as pd
data = pd.read_csv(
    r"C:\Users\Santhos P\OneDrive\Desktop\codesoft\Titanic Survival Prediction\Titanic-Dataset.csv"
)
print("Dataset Loaded Successfully!")
print(data.head())
print("Shape of Dataset:")
print(data.shape)
print("Column Names:")
print(data.columns)
print("Missing Values:")
print(data.isnull().sum())
data = data.drop("PassengerId", axis=1)
data = data.drop("Name", axis=1)
data = data.drop("Ticket", axis=1)
data = data.drop("Cabin", axis=1)
age_median = data["Age"].median()
data["Age"] = data["Age"].fillna(age_median)
embarked_mode = data["Embarked"].mode()[0]
data["Embarked"] = data["Embarked"].fillna(embarked_mode)
print("Data Cleaning Completed!")
print(data.head())
from sklearn.preprocessing import LabelEncoder
sex_encoder = LabelEncoder()
data["Sex"] = sex_encoder.fit_transform(data["Sex"])
embarked_encoder = LabelEncoder()
data["Embarked"] = embarked_encoder.fit_transform(data["Embarked"])
print("After Encoding:")
print(data.head())
X = data.drop("Survived", axis=1)
y = data["Survived"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("Training Data Shape:")
print(X_train.shape)
print("Testing Data Shape:")
print(X_test.shape)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
print("Model Training Completed!")
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:")
print(accuracy)
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)
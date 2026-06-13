import pandas as pd
df = pd.read_csv("creditcard.csv")
print("Dataset Loaded Successfully!")
print(df.head())
print("\nShape:")
print(df.shape)
print("\nColumns:")
print(df.columns.tolist())
print(df['Class'].value_counts())
print(df.isnull().sum())
legit = df[df.Class == 0]
fraud = df[df.Class == 1]
legit_sample = legit.sample(n=492, random_state=42)
new_df = pd.concat([legit_sample, fraud], axis=0)
print(new_df['Class'].value_counts())
X = new_df.drop('Class', axis=1)
y = new_df['Class']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Model Trained Successfully!")
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
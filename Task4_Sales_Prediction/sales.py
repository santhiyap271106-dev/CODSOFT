import pandas as pd
df=pd.read_csv("advertising.csv")
print("Dataset Loaded Successfully!")
print(df.head())
print("\nShape:")
print(df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nMissing Values:")
print(df.isnull().sum())
X=df[['TV','Radio','Newspaper']]
y=df['Sales']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("Training Data:",X_train.shape)
print("Testing Data:",X_test.shape)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)
print("Model Trained Successfully!")
predictions=model.predict(X_test)
print(predictions[:10])
from sklearn.metrics import mean_absolute_error,r2_score
mae=mean_absolute_error(y_test,predictions)
r2=r2_score(y_test,predictions)
print("Mean Absolute Error:",mae)
print("R2 Score:",r2)
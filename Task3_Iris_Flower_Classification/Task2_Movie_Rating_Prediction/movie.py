import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
data = pd.read_csv("IMDb Movies India.csv", encoding="latin1")
data = data.dropna(subset=["Rating"])
data["Votes"] = data["Votes"].str.replace(",", "")
data["Votes"] = pd.to_numeric(data["Votes"])
data["Duration"] = data["Duration"].str.replace(" min", "")
data["Duration"] = pd.to_numeric(data["Duration"])
average_duration = data["Duration"].mean()
data["Duration"] = data["Duration"].fillna(average_duration)
label = LabelEncoder()
data["Genre"] = label.fit_transform(data["Genre"])
data["Director"] = label.fit_transform(data["Director"])
data["Actor 1"] = label.fit_transform(data["Actor 1"])
data["Actor 2"] = label.fit_transform(data["Actor 2"])
data["Actor 3"] = label.fit_transform(data["Actor 3"])
x = data[["Genre", "Director", "Actor 1", "Actor 2", "Actor 3", "Votes", "Duration"]]
y = data["Rating"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(x_train, y_train)
predicted_rating = model.predict(x_test)
mae = mean_absolute_error(y_test, predicted_rating)
r2 = r2_score(y_test, predicted_rating)
print("Mean Absolute Error =", mae)
print("R2 Score =", r2)
plt.scatter(y_test, predicted_rating)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Movie Rating Prediction")
plt.show()
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
train_data = pd.read_csv("fraudTrain.csv",nrows=20000)
test_data = pd.read_csv("fraudTest.csv",nrows=10000)
print("Train Dataset Loaded Successfully!")
print(train_data.head())
train_data = train_data.dropna()
test_data = test_data.dropna()
features = ["amt","city_pop","lat","long","merch_lat","merch_long"]
X_train = train_data[features]
y_train = train_data["is_fraud"]
X_test = test_data[features]
y_test = test_data["is_fraud"]
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n==============================")
print("Model Accuracy :", round(accuracy * 100, 2), "%")
print("==============================")
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report")
print(classification_report(y_test, y_pred))
print("\nEnter Transaction Details")
amt = float(input("Amount: "))
city_pop = int(input("City Population: "))
lat = float(input("Latitude: "))
long = float(input("Longitude: "))
merch_lat = float(input("Merchant Latitude: "))
merch_long = float(input("Merchant Longitude: "))
new_transaction = pd.DataFrame([[amt, city_pop, lat, long, merch_lat, merch_long]],columns=features)
prediction = model.predict(new_transaction)
if prediction[0] == 1:
    print("\nFraudulent Transaction")
else:
    print("\nLegitimate Transaction")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
data = pd.read_csv("train_data.txt",sep=":::",engine="python",names=["ID", "Title", "Genre", "Description"],nrows=15000)
print("Dataset Loaded Successfully!\n")
print(data.head())
data = data.dropna()
data["Genre"] = data["Genre"].str.strip().str.lower()
data["Description"] = data["Description"].str.strip().str.lower()
X = data["Description"]
y = data["Genre"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
tfidf = TfidfVectorizer(stop_words="english",max_features=30000,ngram_range=(1, 2),min_df=2,max_df=0.9,sublinear_tf=True)
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)
model = LinearSVC(C=2.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n==============================")
print("Model Accuracy :", round(accuracy * 100, 2), "%")
print("==============================")
while True:
    movie = input("\nEnter Movie Description : ")
    movie = movie.strip().lower()
    movie_vector = tfidf.transform([movie])
    prediction = model.predict(movie_vector)
    print("\nPredicted Genre :", prediction[0])
    choice = input("\nPredict another movie? (yes/no): ")
    if choice.lower() != "yes":
        print("\nThank You!")
        break
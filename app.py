import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('data/imdb_reviews.csv')

def preprocess_text(text):
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.lower()

df['review'] = df['review'].apply(preprocess_text)

X = df['review']
y = df['sentiment'].map({'positive': 1, 'negative': 0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

st.title("Sentiment Analysis")

num_reviews = st.slider("How many reviews do you want to analyze?", min_value=1, max_value=5, value=1)

reviews = []
for i in range(num_reviews):
    review_input = st.text_area(f"Enter review {i+1}:", key=f"review_{i}")
    reviews.append(review_input)

if st.button("Analyze"):
    if all(reviews):
        predictions = []
        for review in reviews:
            processed_review = preprocess_text(review)
            review_tfidf = tfidf.transform([processed_review])
            prediction = model.predict(review_tfidf)
            sentiment = "Positive" if prediction == 1 else "Negative"
            predictions.append(sentiment)

        for i, sentiment in enumerate(predictions):
            st.write(f"Review {i+1}: {sentiment}")
    else:
        st.write("Please enter all reviews to analyze.")

if st.checkbox("Show Model Performance"):
    y_pred = model.predict(X_test_tfidf)
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.text(classification_report(y_test, y_pred))
    st.text(confusion_matrix(y_test, y_pred))

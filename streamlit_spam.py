import nltk
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    ps = PorterStemmer()

    y = [i for i in text if i.isalnum()]

    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    # Apply stemming
    y = [ps.stem(i) for i in y]

    return " ".join(y)

df = pd.read_csv('modified_spam_data.csv')
df['transformed_text'].fillna("", inplace=True)

X = df['transformed_text']
y = df['v1']

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_model = None
best_accuracy = 0

for train_index, test_index in skf.split(X_vectorized, y):
    X_train, X_test = X_vectorized[train_index], X_vectorized[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Streamlit app interface
st.title("Spam Detection App")
user_input = transform_text(st.text_area("Enter your message:"))

if st.button("Predict"):
    if user_input.strip() != "":
        input_vectorized = vectorizer.transform([user_input])
        prediction = best_model.predict(input_vectorized)[0]

        if prediction == 1:
            st.error("🚨 This message is likely SPAM!")
        else:
            st.success("✅ This message is classified as HAM.")
    else:
        st.warning("Please enter some text.")

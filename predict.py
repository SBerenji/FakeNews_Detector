"""
This file handles loading the saved model and vectorizer 
and provides and function for predictions
"""
import pickle

# Load the trained model and vectorizer
with open('models/fake_news_dectector_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('models/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)


def classify_news(news_article):
    # Vectorize the input article given
    vectorized_input = loaded_vectorizer.transform([news_article])
    # Note: the reason news_article is wrapped in brackets is that transform()
    # expects a list and not a single string

    # Predicting whether the news article is real or fake using the loaded model
    prediction = loaded_model.predict(vectorized_input)

    # prediction is a list that holds the predicted class labels
    # for each of the input samples. Since the input is a single string,
    # we only return the first label
    return prediction[0]

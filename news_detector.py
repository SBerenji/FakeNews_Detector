# The starting point of this project was based on a tutorial from the DataFlair website, with modifications made to the dataset and model.


# importing all the required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# used to convert text data into numerical features
from sklearn.feature_extraction.text import TfidfVectorizer
# used to implement passive-aggressive algorithm, which is a classifier used for binary classification
from sklearn.linear_model import PassiveAggressiveClassifier
# used for evaluating the performance of the model
from sklearn.metrics import accuracy_score, confusion_matrix
# using kaggle dataset
import kagglehub
import os
import pickle  # to save the model

# Download the dataset
dataset_path = kagglehub.dataset_download(
    "clmentbisaillon/fake-and-real-news-dataset")


# Replace with the dataset file name
fake_news_file = os.path.join(dataset_path, "Fake.csv")
real_news_file = os.path.join(dataset_path, "True.csv")


# Checking if files exist
print(f"Fake news file: {fake_news_file}")
print(f"Real news file: {real_news_file}")


# Reading the dataset into dataframes using the pandas library
fake_data = pd.read_csv(fake_news_file)
real_data = pd.read_csv(real_news_file)

# Checking if data was read correctly
print(fake_data.head())
print(real_data.head())

# Adding labels to each dataset
fake_data['label'] = 'FAKE'
real_data['label'] = 'REAL'

# Merging the datasets into a single dataframe
df = pd.concat([fake_data, real_data], ignore_index=True)

# Shuffling the dataset to ensure randomness
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

# Getting the labels from the dataframe
labels = df.label

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    df['text'], labels, test_size=0.2, random_state=7)

# df['text'] is the feature set, the model will use this text data to learn patterns (get trained)
# labels extracted previously is the target variable (fake or real news) which the model will predict
# test size is set to 0.2 which means 20% of the data will be used for testing and 80% will be used for training
# the seed (random_state) is set to 7 so we can get the same training and testing sets when using this seed


# Initialzing the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english', max_df=0.7, ngram_range=(1, 2))
# Based on stop_words='english, the common english words like 'and' or 'the' will not be considered when trying to distinguish if the news is fake or real
# Based on max_df=0.7, words that appear in more than 70% of the documents will be ignored.
# (because words with high document frequency (e.g., "news," "article") are less informative)
# By using ngram_range=(1, 2) the vectorizer considers both unigrams (individual words) and bigrams (two consecutive words) as features

# Fit and transform train set and test set
# fit_transform function first learns the parameters from the data and then transforms the data into its numerical representation.
# The vocabulary (set of unique words) and their corresponding IDF scores are learned from the training data and stored in the vectorizer object.
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
# transform function applies the previously learned parameters (from fit_transform) to new data without re-learning them
# Here the vectorizer does not learn anything new. It applies the same vocabulary and IDF values to transform x_test into a numerical matrix.
# For words in the test data that are in the vocabulary, their TF-IDF scores are computed using the IDF values from the training data.
tfidf_test = tfidf_vectorizer.transform(x_test)

# Initializing a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50, C=0.5)
pac.fit(tfidf_train, y_train)

# Predicting on the test set and calculating the accuracy of the model
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {score}")

# Printing a confusion matrix
# The result will show how many of the news were correctly predicted as fake or real
# first row is for the ones labeled as 'fake' and second row is for 'real' label
print(confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']))

# Saving the trained model
with open('models/fake_news_dectector_model.pkl', 'wb') as model_file:
    pickle.dump(pac, model_file)

with open('models/tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

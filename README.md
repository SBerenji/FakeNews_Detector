
# Fake News Detector




## Project Overview

The goal of this project is to predict whether a news article is fake or real based on its content. The model is trained using a dataset from Kaggle, which contains two separate CSV files: one for fake news and another for real news.



## Dataset
The dataset is sourced from Kaggle, and it consists of two CSV files:

- Fake.csv: Contains news articles labeled as fake.
- True.csv: Contains news articles labeled as real.
These files are used to train and evaluate the model.
## Packages Used

- numpy
- pandas
- scikit-learn
- kagglehub
- jupyter

## Requirements

To run this project, you will need to install the required Python packages. You can do this by creating a virtual environment and installing the dependencies from the requirements.txt file by running the "pip install -r requirements.txt" command.



## Model Details
- Model Type: Passive Aggressive Classifier
- Features: The text content of the news articles is used as the input features. The text is transformed into numerical features using TF-IDF Vectorization.
- Accuracy: The model achieves an accuracy of approximately 99.43% on the test data.

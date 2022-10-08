#!/usr/bin/env python
# coding: utf-8

# Import
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Load data
trainingSet = pd.read_csv("train.csv")
testingSet = pd.read_csv("test.csv")
print("finish loading")


# Lemmatization text by removing stopwords 
def lemmatization(text):
    stopWord = stopwords.words('english')
    # init Lemmatizer
    lemmatizer = WordNetLemmatizer()
    # remove non charaters
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    # change all charaters to lower case
    words = text.lower().split()
    # run lemmatizer
    words = [lemmatizer.lemmatize(w) for w in words if w not in stopWord]
    return ' '.join(words)


# Adding new column of clean_review
trainingSet['clean_review'] = [lemmatization(text) for text in trainingSet.Text]
print("finsih lemmatization, see detail below:")
print(trainingSet)


# create x_text and x_train
X_test = pd.merge(trainingSet, testingSet, left_on='Id', right_on='Id')
print(X_test)

# Code provided in generate-Xtrain-Xsubmission.py
X_test = X_test.drop(columns=['Score_x'])
X_test = X_test.rename(columns={'Score_y': 'Score'})
X_test.to_csv("X_submission.csv", index=False)
X_train = trainingSet[trainingSet['Score'].notnull()]
X_train.to_csv("X_train.csv", index=False)
print("finish X_train and X_submission split")

print(X_train)

# Read from file
X_train = pd.read_csv("X_train.csv")
X_submission = pd.read_csv("X_submission.csv")

# Split training set into training and testing set
# Code provided in predict-knn.py
X_train, X_test, Y_train, Y_test = train_test_split(
    X_train.drop(['Score'], axis=1),
    X_train['Score'],
    test_size=1 / 4.0,
    random_state=0
)

X_train['clean_review'] = X_train['clean_review'].fillna("a")
X_test['clean_review'] = X_test['clean_review'].fillna("a")
X_submission['clean_review'] = X_submission['clean_review'].fillna("a")

# Method 1: Predict with vectorlize clean_review
# inti by setting feature number
vectorizer = CountVectorizer(max_features=8000)

# Vectorlize X_train, X_test and X_Submission
vector_train = vectorizer.fit_transform(X_train.clean_review).toarray()
vector_test = vectorizer.transform(X_test.clean_review).toarray()
vector_submit = vectorizer.transform(X_submission.clean_review).toarray()

# Learn the review model
# Create random forest model
forestor = RandomForestClassifier(n_estimators=100, n_jobs=8)
# Set Grid Search parameters
n_estimators = [100]
min_samples_split = [2]
min_samples_leaf = [1]
bootstrap = [True]
parameters = {'n_estimators': n_estimators, 'min_samples_leaf': min_samples_leaf,
              'min_samples_split': min_samples_split}
# Apply Grid search 
clf = GridSearchCV(forestor, param_grid=parameters)
# Fit data to creat model
model_vector = clf.fit(vector_train, Y_train)
print("finish model_review")

# Method 2: Regular predict with data frame
# Process the DataFrames
# This is where you can do more feature extraction
X_train_processed = X_train.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'clean_review'])
X_test_processed = X_test.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'clean_review'])
X_submission_processed = X_submission.drop(
    columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'Score', 'clean_review'])

# Learn the review model
# create random forest model
forestor2 = RandomForestClassifier(n_estimators=100, n_jobs=8)
# Set Grid Search parameters
n_estimators = [100]
min_samples_split = [2]
min_samples_leaf = [1]
bootstrap = [True]
parameters = {'n_estimators': n_estimators, 'min_samples_leaf': min_samples_leaf,
              'min_samples_split': min_samples_split}
# Apply Grid search 
clf2 = GridSearchCV(forestor2, param_grid=parameters)
# Fit data to creat model
model = clf2.fit(X_train_processed, Y_train)
print("finish model")

# Start predict
# Predict using vector
Y_test1 = model_vector.predict(vector_test)
X_submission['Score1'] = model_vector.predict(vector_submit)

# Predict using regular pd
Y_test2 = model.predict(X_test_processed)
X_submission['Score2'] = model.predict(X_submission_processed)

# Combine two methods together with different weight
Y_av = (Y_test1 * 0.5 + Y_test2 * 0.5).round(0)
X_submission['Score'] = (X_submission['Score1'] * 0.5 + X_submission['Score2'] * 0.5).round(0)

# Evaluate your model on the testing set
print("RMSE on testing set: review predict = ", mean_squared_error(Y_test, Y_test1))
print("RMSE on testing set: regular predict = ", mean_squared_error(Y_test, Y_test2))
print("RMSE on testing set: regular predict = ", mean_squared_error(Y_test, Y_av))

# Create the submission file
submission = X_submission[['Id', 'Score']]
submission.to_csv("submission.csv", index=False)

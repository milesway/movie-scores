# Import Packages

import re
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix
import nltk
#nltk.download()
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfTransformer

# Starter Code README

#Run initial-exploration.py to print the first few rows of train.csv and test.csv and view their shapes + some visualization
#Run generate-Xtrain-Xsubmission.py to generate X_train.csv which you will use to learn the model and X_submission.csv which you will use to apply your model and generate your predictions for grading
#Run predict-constant.py to predict the same score for all rows in the testing set
#Run predict-knn.py to predict the score using KNN

#load data
trainingSet = pd.read_csv("train.csv")
testingSet = pd.read_csv("test.csv")

#add more features

#1. add average user score and average product score
#1.1 user average score
trainingSet['Score']=trainingSet['Score'].fillna(3)
train_user = pd.DataFrame(trainingSet[['UserId','Score']])
train_user_av = train_user.groupby(['UserId']).mean().round(0)
train_user_avscore = train_user_av.rename(columns = {"Score":"User_average"})

result = pd.merge(trainingSet, train_user_avscore, how="left",on="UserId")
#1.2 product average score
train_product = pd.DataFrame(trainingSet[['ProductId','Score']])
train_product_av = train_product.groupby(['ProductId']).mean().round(0)
train_product_avscore = train_product_av.rename(columns = {"Score":"Product_average"})

trainingSet = pd.merge(result, train_product_avscore, how="left",on="ProductId")
trainingSet['User_average']=trainingSet['User_average'].fillna(3)
trainingSet['Product_average']=trainingSet['Product_average'].fillna(3)
print("finish user average and product average")

#2. add lemma review
#clean words with stopwords and delete all not words and tokenize
#lemmatization
STOPWORDS = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def lemmatization(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    words = text.lower().split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in STOPWORDS]
    return ' '.join(words)

trainingSet['lemma_review'] = trainingSet.Text.apply(lemmatization)
print("finish lemma review")
trainingSet['lemma_sum'] = trainingSet.Summary.apply(lemmatization)
print("finish lemma review and summary on training set")

#3. add polarity
#do sentiment test
def sentiment(text):
    textblob = TextBlob(text)
    return round(textblob.polarity,2)    

trainingSet['Polarity_review'] = trainingSet.lemma_review.apply(sentiment)
trainingSet['Polarity_sum'] = trainingSet.lemma_sum.apply(sentiment)
print("finish polarity of review and summary on training set")

#4. vecotorize lemma review and summary
#vectorize train set on review
#vectorizer_review = CountVectorizer(max_features = 3000) 
#trainingCounts = vectorizer_review.fit_transform(trainingSet.lemma_review)

#from occurance to frequencies
#tfidf_transformer_review = TfidfTransformer()
#tfidf = tfidf_transformer_review.fit_transform(trainingCounts).toarray()

#vectorize train set on summary
#vectorizer_sum = CountVectorizer(max_features = 1000) 
#trainingSet['vector_sum'] = vectorizer_sum.fit_transform(trainingSet.lemma_sum)

#from occurance to frequencies
#tfidf_transformer_sum = TfidfTransformer()
#trainingSet['tfidf_sum'] = tfidf_transformer_sum.fit_transform(trainingSet['vector_sum'])

print("finish tfidf on review and summary of training set")

#create x_text and x_train
X_test = pd.merge(trainingSet, testingSet, left_on='Id', right_on='Id')

X_test = X_test.drop(columns=['Score_x'])
X_test = X_test.rename(columns={'Score_y': 'Score'})

X_test.to_csv("X_submission.csv", index=False)

X_train = trainingSet[trainingSet['Score'].notnull()]

X_train.to_csv("X_train.csv", index=False)

# Load files into DataFrames
X_train = pd.read_csv("X_train.csv")
X_submission = pd.read_csv("X_submission.csv")

# Split training set into training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(
        X_train.drop(['Score'], axis=1),
        X_train['Score'],
        test_size=1/4.0,
        random_state=0
    )

# Process the DataFrames
# This is where you can do more feature extraction
X_train_processed = X_train.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'lemma_review','lemma_sum'])
X_test_processed = X_test.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary','lemma_review','lemma_sum'])
X_submission_processed = X_submission.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'Score','lemma_review','lemma_sum'])

# Learn the model
#create random forest model
forest = RandomForestClassifier(n_estimators = 100)
model = forest.fit(X_train_processed, Y_train)
print("finish fit easy one")

# Predict the score using the model
Y_test_predictions = model.predict(X_test_processed)
X_submission['Score'] = model.predict(X_submission_processed)

# Evaluate your model on the testing set
print("RMSE on testing set = ", mean_squared_error(Y_test, Y_test_predictions))

# Plot a confusion matrix
cm = confusion_matrix(Y_test, Y_test_predictions, normalize='true')
print(cm)
#sns.heatmap(cm, annot=True)
#plt.title('Confusion matrix of the classifier')
#plt.xlabel('Predicted')
#plt.ylabel('True')
#plt.show()

# Create the submission file
submission = X_submission[['Id', 'Score']]
submission.to_csv("submission.csv", index=False)

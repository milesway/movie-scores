# Import Packages

import re
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from collections import Counter

#load data
trainingSet = pd.read_csv("train.csv")
testingSet = pd.read_csv("test.csv")

#add more features

#1. add average user score and average product score
#1.1 user average score
train_user = pd.DataFrame(trainingSet[['UserId','Score']])
train_user_av = train_user.groupby(['UserId']).mean().round(0)
train_user_avscore = train_user_av.rename(columns = {"Score":"User_average"})

train_user_avscore['User_average'] = train_user_avscore['User_average'].fillna(3)

result = pd.merge(trainingSet, train_user_avscore, how="left",on="UserId")

#1.2 product average score
train_product = pd.DataFrame(trainingSet[['ProductId','Score']])
train_product_av = train_product.groupby(['ProductId']).mean().round(0)
train_product_avscore = train_product_av.rename(columns = {"Score":"Product_average"})

train_product_avscore['Product_average'] = train_product_avscore['Product_average'].fillna(3)

trainingSet = pd.merge(result, train_product_avscore, how="left",on="ProductId")
print(trainingSet['Product_average'].isnull().sum())
print(trainingSet['User_average'].isnull().sum())
print("finish user average and product average")

#2. use nltk to find conpound sentiment score
sia = SentimentIntensityAnalyzer()
def nltk_sentiment(text):
    text = str(text)
    return sia.polarity_scores(text).get('compound')

trainingSet['Sentiment_nltk'] = trainingSet.Text.apply(nltk_sentiment)
print("finish nltk sentiment")

#2. add lemma review
#clean words with stopwords and delete all not words and tokenize
#lemmatization
STOPWORDS = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
pos_list = ['NN','RB','VB','JJ','VBP','PDT','RBR','RBS','UH','VBG','VBD','VBN','VBP','JJR','JJS']

def lemmatization(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    words = text.lower().split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in STOPWORDS]
    words = [word for word,pos in pos_tag(words) if pos in pos_list]
    return ' '.join(words)

trainingSet['lemma_review'] = trainingSet.Text.apply(lemmatization)
#trainingSet['lemma_sum'] = trainingSet.Summary.apply(lemmatization)
print("trainingSet lemma review")
print(trainingSet['lemma_review'].isnull().sum())
print("finish lemma review and summary on training set")

#3. add polarity
#do sentiment test
#def sentiment(text):
    #textblob = TextBlob(text)
    #return round(textblob.polarity,3)   

#trainingSet['Polarity_review'] = trainingSet.lemma_review.apply(sentiment)
#trainingSet['Polarity_sum'] = trainingSet.lemma_sum.apply(sentiment)
#print("finish polarity of review and summary on training set")


#4.1 find most common 50 words in each score
train_word = pd.DataFrame(trainingSet[['lemma_review','Score']])
train_word = train_word.dropna()
common_word = train_word.groupby(['Score'])
s1_list = []
s2_list = []
s3_list = []
s4_list = []
s5_list = []
for name, group in common_word:
    if name == 1:
        string = ' '.join(group.lemma_review)
        s1_list.append([word for word, count in Counter(string.split()).most_common(50)])
    elif name == 2:
        string = ' '.join(group.lemma_review)
        s2_list.append([word for word, count in Counter(string.split()).most_common(50)])
    elif name == 3:
        string = ' '.join(group.lemma_review)
        s3_list.append([word for word, count in Counter(string.split()).most_common(50)])
    elif name == 4:
        string = ' '.join(group.lemma_review)
        s4_list.append([word for word, count in Counter(string.split()).most_common(50)])
    elif name == 5:
        string = ' '.join(group.lemma_review)
        s5_list.append([word for word, count in Counter(string.split()).most_common(50)])

cw = pd.DataFrame([s1_list, s2_list, s3_list, s4_list, s5_list],columns=[1,2,3,4,5])
print(cw)
print("finish top 50 words")

#create x_text and x_train
X_test = pd.merge(trainingSet, testingSet, left_on='Id', right_on='Id')
print("finish merge")

X_test = X_test.drop(columns=['Score_x'])
X_test = X_test.rename(columns={'Score_y': 'Score'})

X_test.to_csv("X_submission.csv", index=False)

X_train = trainingSet[trainingSet['Score'].notnull()]
print("x_train lemma review")
print(X_train['lemma_review'].isnull().sum())

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

print("start vector")
X_train['lemma_review'] = X_train['lemma_review'].fillna("n")
X_test['lemma_review'] = X_test['lemma_review'].fillna("n")
X_submission['lemma_review'] = X_submission['lemma_review'].fillna("n")
#4.2 guess the score only using frequent word lists

#5. vecotorize lemma review and summary
#vectorize train set on review
vectorizer_review = CountVectorizer(max_features = 5000) 
vector_review = vectorizer_review.fit_transform(X_train.lemma_review)
#from occurance to frequencies
tfidf_transformer_review = TfidfTransformer()
train_tfidf_review = tfidf_transformer_review.fit_transform(vector_review).toarray()

#vectorize test set on test review
test_vector_review = vectorizer_review.transform(X_test.lemma_review)
#from occurance to frequencies
test_tfidf_review = tfidf_transformer_review.transform(test_vector_review).toarray()

#vectorize submission set on submission review
sub_vector_review = vectorizer_review.transform(X_submission.lemma_review)
#from occurance to frequencies
sub_tfidf_review = tfidf_transformer_review.transform(sub_vector_review).toarray()

print("finish tfidf on review and summary of training set")

# Process the DataFrames
# This is where you can do more feature extraction
X_train_processed = X_train.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'lemma_review'])
X_test_processed = X_test.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary','lemma_review'])
X_submission_processed = X_submission.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'Score','lemma_review'])

# Learn the review model
#create random forest model
forest_review = RandomForestClassifier(n_estimators = 100, n_jobs=4)
model_review = forest_review.fit(train_tfidf_review, Y_train)
print("finish review fit")

# Learn the model
#create random forest model
forest = RandomForestClassifier(n_estimators = 100, n_jobs=4)
model = forest.fit(X_train_processed, Y_train)
print("finish regular fit")

# Predict based on review
Y_test_review_pre = model_review.predict(test_tfidf_review)
X_submission['Score1'] = model_review.predict(sub_tfidf_review)

# Predict the score using the model
Y_test_regular_pre = model.predict(X_test_processed)
X_submission['Score2'] = model.predict(X_submission_processed)

# try to find the best combination
#0.5:0.5
Y_av3 = (Y_test_review_pre * 0.5 + Y_test_regular_pre * 0.5).round(0)
X_submission['Score3']=(X_submission['Score1']*0.5+X_submission['Score2']*0.5).round(0)
#4:6
Y_av4 = (Y_test_review_pre * 0.4 + Y_test_regular_pre * 0.6).round(0)
X_submission['Score4']=(X_submission['Score1']*0.4+X_submission['Score2']*0.6).round(0)
#3:7
Y_av5 = (Y_test_review_pre * 0.3 + Y_test_regular_pre * 0.7).round(0)
X_submission['Score5']=(X_submission['Score1']*0.3+X_submission['Score2']*0.7).round(0)
#2:8
Y_av6 = (Y_test_review_pre * 0.2 + Y_test_regular_pre * 0.8).round(0)
X_submission['Score6']=(X_submission['Score1']*0.2+X_submission['Score2']*0.8).round(0)

# Evaluate your model on the testing set
print("RMSE on testing set: review predict = ", mean_squared_error(Y_test, Y_test_review_pre))
print("RMSE on testing set: regular predict = ", mean_squared_error(Y_test, Y_test_regular_pre))
print("RMSE on testing set: regular predict = ", mean_squared_error(Y_test, Y_av3))
print("RMSE on testing set: regular predict = ", mean_squared_error(Y_test, Y_av4))
print("RMSE on testing set: regular predict = ", mean_squared_error(Y_test, Y_av5))
print("RMSE on testing set: regular predict = ", mean_squared_error(Y_test, Y_av6))

# Plot a confusion matrix
#cm1 = confusion_matrix(Y_test, Y_test_review_pre, normalize='true')
#print(cm1)
#cm2 = confusion_matrix(Y_test, Y_test_regular_pre, normalize='true')
#print(cm2)
#sns.heatmap(cm, annot=True)
#plt.title('Confusion matrix of the classifier')
#plt.xlabel('Predicted')
#plt.ylabel('True')
#plt.show()

# Create the submission file
submission = X_submission[['Id', 'Score1']]
submission.to_csv("submission1.csv", index=False)

submission = X_submission[['Id', 'Score2']]
submission.to_csv("submission2.csv", index=False)

submission = X_submission[['Id', 'Score3']]
submission.to_csv("submission3.csv", index=False)

submission = X_submission[['Id', 'Score4']]
submission.to_csv("submission4.csv", index=False)

submission = X_submission[['Id', 'Score5']]
submission.to_csv("submission5.csv", index=False)

submission = X_submission[['Id', 'Score6']]
submission.to_csv("submission6.csv", index=False)


import pandas as pd
import matplotlib.pyplot as plt

# packages for the ML model

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, auc

# packages for modifying the data set

from nltk.corpus import stopwords
from collections import Counter

# import data

df = pd.read_csv('new_data.csv')

df['extract'] = df['extract'].fillna(' ')

# create a dummy data set
# this will allow us to make modifications to the data frame in different ways with a single run of code

dum = df

# check how many scores are positive vs negative

print(pd.crosstab(index = df['positive'], columns="Total count"))

# 436738 positive, 111032 negative, which shows most of the reviews are positive
# so it is important that when building our model to balance the class weights

# creation of the machine learning model

# tokenization

cvec = CountVectorizer()

def pred_model(df):

    # split data into training and test sets

    X_train, X_test, y_train, y_test = train_test_split(df['extract'], df['positive'], test_size=0.3, random_state=0)

    # Fit the vectorizer to the training data

    vect = cvec.fit(X_train)

    # transform the documents in the training data to a document-term matrix

    X_train_vectorized = vect.transform(X_train)

    model = LogisticRegression(max_iter=len(X_train), class_weight='balanced')
    model.fit(X_train_vectorized, y_train)

    # checking fitness of model

    predictions = model.predict(vect.transform(X_test))
    aruc = roc_auc_score(y_test, predictions)
    print('AUC: ', aruc)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    # plot AUC

    plt.figure(figsize=(10, 8), dpi=100)
    plt.axis('scaled')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("AUC & ROC Curve")
    plt.plot(false_positive_rate, true_positive_rate, 'g')
    plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

pred_model(df)

# AUC of the base model is 0.85
# this means that our predictive model is fairly accurate, but it can possibly be more accurate
# since count vectorizer looks solely at the words, overlap of common words
# including ones that are regular stopwords or ones specific to phones like "battery", "screen", or "new"
# thus, we can create a more clear set of data

print(Counter(" ".join(dum["extract"]).split()).most_common(100))

# as one might expect, the most common words are largely stopwords, which might affect the model

stop = stopwords.words('english')

dum['extract'] = df['extract'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

pred_model(dum)

# it appears that removing the stopwords made the model slightly worse, but it may be worth looking into
# the other possible set of stopwords

print(Counter(" ".join(dum['extract']).split()).most_common(100))

# this set of the 100 most common words after removing generic stopwords gives many words that were to be expected
# such as "phone", "screen", "camera" and "battery"
# choosing words to omit to potentially improve the model may be tricky and up to personal beliefs as to which ones
# would be most common

new_stop = ['I', 'phone', 'The', 'phone.', 'It', 'This', 'battery', 'one', 'use', 'screen', 'it.', 'camera', 'really',
            'get', 'phone,', 'got', 'would', 'bought', 'quality', 'new', 'phones', 'Samsung', 'using', 'price',
            'mobile', "I've", 'first', 'buy', '-', 'features', 'much', 'product',  'time', '2', 'used', 'Very', "I'm",
            "It's", 'back', 'also', 'life', 'still', 'well', 'Phone', 'My', 'ever', '.', 'everything', 'want', 'need',
            'iPhone', 'many', '3', 'fast', 'could', 'apps', ',', 'last', 'Galaxy', 'go', 'say', 'old', 'But', 'two',
            'recommend', 'came', 'never', 'thing', 'lot', 'day', 'since', 'card', 'A', 'touch', '&', 'days', '5',
            'looks']

stop.extend(new_stop)

dum['extract'] = df['extract'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

pred_model(dum)

# the AUC is lower again, which means that the original model that contained all the stopwords was the most accurate
# this is not an unexpected situation, as it sometimes happens with stopwords
# one more model will be tried, removing only the 10 most common words in English

stop = ['the', 'of', 'to', 'and', 'a', 'in', 'is', 'it', 'you', 'that']

dum['extract'] = df['extract'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

pred_model(dum)

# this one is still slightly less accurate than the full model, so in this data set, stop words increase the accuracy

# though it may also be possible to improve the model by removing words that appear too infrequently and improving
# the ngram range so that words like "not good" will register

cvec = CountVectorizer(min_df=5, ngram_range=(1,2))

pred_model(df)

# this did improve the model slightly

cvec = CountVectorizer(ngram_range=(1,2))

pred_model(df)

# this model that does not remove any words from the reviews improved the model slightly more
# perhaps increasing the ngram range a little further could also be useful

cvec = CountVectorizer(ngram_range=(1,3))

pred_model(df)

# this is also slightly better but it is only about a 0.01% improvement, so ngram will not be explored further

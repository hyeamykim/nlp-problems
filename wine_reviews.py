
# coding: utf-8

"""
disclaimer: The following codes were studied and used for this task

Multi-Class Text Classification with Scikit-Learn
https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f

NLP + feature based points classification
https://www.kaggle.com/mistrzuniu1/nlp-feature-based-points-predictions/

How to use Machine Learning to Predict the Quality of Wines
https://medium.freecodecamp.org/using-machine-learning-to-predict-the-quality-of-wines-9e2e13d7480d

Last updated: 25.08.2018
By: Hye Yeon Kim
"""


# Part 1: Cleaning and preparing the data
# importing necessary libraries

import pandas as pd
import numpy as np

import string
import nltk 
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')

# importing data
data = pd.read_csv('wine-reviews\winemag-data-130k-v2.csv')

# checking how data looks like
data.head()

# dropping unnecessary unnamed columns
data=data.drop('Unnamed: 0', axis=1)
data=data.reset_index(drop=True)

# checking for duplicates
print("Total number of examples: ", data.shape[0])
print("Number of examples with the same title and description: ", data[data.duplicated(['description','title'])].shape[0])

# dropping duplicate wine reviews
data=data.drop_duplicates(['description','title'])
data=data.reset_index(drop=True)

# checking how much of data is missing
total = data.isnull().sum().sort_values(ascending = False)
percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
missing  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing

# dropping NA values in our target, taster_name
data=data.dropna(subset=['taster_name'])
data=data.reset_index(drop=True)


# Part 2: Exploratory Data Analysis
# adding description length in the data
data['length'] = data['description'].apply(len)

# checking the distrbution of description length
data['length'].plot(bins=50, kind='hist')

# checking if there is visible difference of description length with each taster
data.hist(column='length', by='taster_name', bins=50, figsize=(10,4))

# checking how many descriptions are there per each taster
fig = plt.figure(figsize=(8,6))
data.groupby('taster_name').description.count().plot.bar(ylim=0)


# Part 3: Natural Language Processing
# preparing data for training by creating a smaller dataframe and by giving id to each taster

from io import StringIO

# creating a new data frame of taster_name and description
col = ['taster_name', 'description']
df = data[col]
df = df[pd.notnull(df['description'])]
df = df[pd.notnull(df['taster_name'])]
df.columns = ['taster_name', 'description']

# giving id to each taster
data['taster_id'] = data['taster_name'].factorize()[0]
taster_id_df = data[['taster_name', 'taster_id']].drop_duplicates().sort_values('taster_id')
taster_to_id = dict(taster_id_df.values)
id_to_taster = dict(taster_id_df[['taster_id', 'taster_name']].values)

# checking how it looks
data.head()

# importing more libraries for language processing

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS
import re
from nltk.tokenize import RegexpTokenizer

# for text preprocessing

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)  
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# checking if text_process works
data.description.head(5).apply(text_process)


# Part 4: Training and Testing the data
# creating a pipeline to fit data
pipeline = Pipeline([('bow', CountVectorizer(analyzer=text_process)),
                    ('tfidf', TfidfTransformer()),
                     ('classifier', MultinomialNB())                 
                    ])

# splitting the data for training set and test set
dat_train, dat_test, name_train, name_test = train_test_split(data['description'], data['taster_name'], test_size =0.2)

# training the model
pipeline.fit(dat_train, name_train)

# predicting with the model
predictions = pipeline.predict(dat_test)

# prediction results
print(classification_report(predictions, name_test))


# Part 5: Different approaches of training and testing the data
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostRegressor, cv

# feature extraction for further usage of different models
vect = TfidfVectorizer(analyzer='word', stop_words ='english', ngram_range=(1, 2), token_pattern=r'\w+',max_features=500)
features = vect.fit_transform(data['description']).toarray()
labels = data.taster_id
features.shape

# checking related features for each taster
from sklearn.feature_selection import chi2

N = 2
for taster_name, taster_id in sorted(taster_to_id.items()):
  features_chi2 = chi2(features, labels == taster_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(vect.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(taster_name))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

# preparing to compare different models

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []

# implementing different models and appending the accuracy results
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))

# creating a dataframe of accuracy of different models used
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
cv_df.head()

# visualizing the accuracy of different models
import seaborn as sns
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()

# comparing accuracy of prediction results of different models
cv_df.groupby('model_name').accuracy.mean()

# visualizing confusion matrix to check the prediction results
model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, data.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=taster_id_df.taster_name.values, yticklabels=taster_id_df.taster_name.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# Comments:
# 
# 1. I did try using other features in the reviews table, but decided to use only description as a feature, to focus on natural language processing.
# 2. In part 5, trying with other models for training and testing the data, I only used Tfidf feature extraction to make the process simpler, instead of using a bit more extensive pipeline.
# 3. I know it takes too long to run the code, but since I am a novice in pyspark, tensorflow, and keras (all of which I would have liked to use), I could not make it work, so I just chose what worked in the end.
# 
# Next Steps (personal agenda):
# 
# 1. Hyperparameter fine-tuning with GridSearchCV
# 2. Changing the code to apply it to pyspark
# 3. Using Tensorflow, Keras for faster process 

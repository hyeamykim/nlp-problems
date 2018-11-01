
# coding: utf-8

import nltk # Imports the library

messages = [line.rstrip() for line in open ('SMSSpamCollection', 'rt', encoding='UTF8')]
print(len(messages))

for num, message in enumerate (messages[:10]):
    print(num, message)
    print('\n')

import pandas

messages = pandas.read_csv('SMSSpamCollection', 
                           sep='\t', names=['label','message'])
messages.head()

messages.describe()

messages.info()

messages.groupby('label').describe()

messages['length'] = messages['message'].apply(len)
messages.head()


import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

messages['length'].plot(bins=50, kind='hist')

messages['length'].describe()

messages[messages['length']==910]['message'].iloc[0]

messages.hist(column='length', by='label', bins=50, figsize=(10,4))

import string

mess = 'Sample message! Notice: it has punctuation'

nopunc = [char for char in mess if char not in string.punctuation]

nopunc

nopunc = ''.join(nopunc)

nopunc

from nltk.corpus import stopwords

stopwords.words('English')[0:10]

nopunc.split()


clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

clean_mess

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

messages.head()

messages['message'].head(5).apply(text_process)

from sklearn.feature_extraction.text import CountVectorizer

bow_transformer = CountVectorizer(analyzer=text_process)

bow_transformer.fit(messages['message'])

message4 = messages['message'][3]

print(message4)

bow4 = bow_transformer.transform([message4])

print(bow4)

print(bow_transformer.get_feature_names()[9554])


messages_bow = bow_transformer.transform(messages['message'])

print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)
print('sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])))

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)

tfidf4 = tfidf_transformer.transform(bow4)

print(tfidf4)

print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])

print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])

messages_tfidf = tfidf_transformer.transform(messages_bow)

print(messages_tfidf.shape)

from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])

print('predicted: ', spam_detect_model.predict(tfidf4)[0])
print('expected: ', messages['label'][3])

all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)

from sklearn.metrics import classification_report
print(classification_report(messages['label'], all_predictions))

from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size =0.2)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))

from sklearn.pipeline import Pipeline

pipeline = Pipeline([('bow', CountVectorizer(analyzer=text_process)),
                    ('tfidf', TfidfTransformer()),
                     ('classifier', MultinomialNB())                 
                    ])

pipeline.fit(msg_train, label_train)

predictions = pipeline.predict(msg_test)

print(classification_report(predictions, label_test))


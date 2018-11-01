
# coding: utf-8

# In[1]:


# ONLY RUN THIS CELL IF YOU NEED 
# TO DOWNLOAD NLTK AND HAVE CONDA

# Uncomment the code below and run:
import nltk # Imports the library
# nltk.download() #Download the necessary datasets


# In[3]:


pwd


# In[2]:


messages = [line.rstrip() for line in open ('SMSSpamCollection', 'rt', encoding='UTF8')]


# In[3]:


print(len(messages))


# In[4]:


for num, message in enumerate (messages[:10]):
    print(num, message)
    print('\n')


# In[5]:


import pandas


# In[6]:


messages = pandas.read_csv('SMSSpamCollection', 
                           sep='\t', names=['label','message'])


# In[7]:


messages.head()


# In[8]:


messages.describe()


# In[9]:


messages.info()


# In[10]:


messages.groupby('label').describe()


# In[11]:


messages['length'] = messages['message'].apply(len)
messages.head()


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')


# In[13]:


messages['length'].plot(bins=50, kind='hist')


# In[14]:


messages['length'].describe()


# In[15]:


messages[messages['length']==910]['message'].iloc[0]


# In[16]:


messages.hist(column='length', by='label', bins=50, figsize=(10,4))


# In[17]:


import string


# In[18]:


mess = 'Sample message! Notice: it has punctuation'


# In[19]:


nopunc = [char for char in mess if char not in string.punctuation]


# In[20]:


nopunc


# In[21]:


nopunc = ''.join(nopunc)


# In[22]:


nopunc


# In[23]:


from nltk.corpus import stopwords


# In[24]:


stopwords.words('English')[0:10]


# In[25]:


nopunc.split()


# In[26]:


clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[27]:


clean_mess


# In[28]:


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


# In[29]:


messages.head()


# In[30]:


messages['message'].head(5).apply(text_process)


# In[31]:


from sklearn.feature_extraction.text import CountVectorizer


# In[33]:


bow_transformer = CountVectorizer(analyzer=text_process)


# In[34]:


bow_transformer.fit(messages['message'])


# In[35]:


message4 = messages['message'][3]


# In[36]:


print(message4)


# In[39]:


bow4 = bow_transformer.transform([message4])


# In[40]:


print(bow4)


# In[42]:


print(bow_transformer.get_feature_names()[9554])


# In[43]:


messages_bow = bow_transformer.transform(messages['message'])


# In[45]:


print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)
print('sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])))


# In[46]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[47]:


tfidf_transformer = TfidfTransformer().fit(messages_bow)


# In[49]:


tfidf4 = tfidf_transformer.transform(bow4)


# In[50]:


print(tfidf4)


# In[51]:


print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])


# In[52]:


print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])


# In[53]:


messages_tfidf = tfidf_transformer.transform(messages_bow)


# In[54]:


print(messages_tfidf.shape)


# In[55]:


from sklearn.naive_bayes import MultinomialNB


# In[57]:


spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])


# In[60]:


print('predicted: ', spam_detect_model.predict(tfidf4)[0])
print('expected: ', messages['label'][3])


# In[61]:


all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)


# In[62]:


from sklearn.metrics import classification_report
print(classification_report(messages['label'], all_predictions))


# In[63]:


from sklearn.model_selection import train_test_split


# In[64]:


msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size =0.2)


# In[65]:


print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))


# In[67]:


from sklearn.pipeline import Pipeline


# In[68]:


pipeline = Pipeline([('bow', CountVectorizer(analyzer=text_process)),
                    ('tfidf', TfidfTransformer()),
                     ('classifier', MultinomialNB())                 
                    ])


# In[69]:


pipeline.fit(msg_train, label_train)


# In[70]:


predictions = pipeline.predict(msg_test)


# In[73]:


print(classification_report(predictions, label_test))


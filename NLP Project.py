#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[9]:


import pandas as pd # data preprocessing
import itertools # confusion matrix
import string
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# To show all the rows of pandas dataframe
pd.set_option('display.max_rows', None)


# # Importing the Dataset

# In[10]:


df = pd.read_csv('/Users/akshitkamboj/Downloads/drugsCom_raw/drugsComTrain_raw.tsv', sep='\t')


# In[127]:


df.to_csv('NLP_Dataset.csv')


# # Studying the Dataset

# In[3]:


df.head()


# In[4]:


df.condition.value_counts()


# In[11]:


df_train = df[(df['condition']=='Birth Control') | (df['condition']=='Depression') | (df['condition']=='High Blood Pressure')|(df['condition']=='Diabetes, Type 2')]


# In[7]:


df.shape


# In[8]:


df_train.shape


# In[12]:


X = df_train.drop(['Unnamed: 0','drugName','rating','date','usefulCount'],axis=1)


# In[10]:


X.head()


# In[12]:


df_train.head()


# # Exploratory Data Analysis

# In[13]:


X.condition.value_counts()


# In[14]:


X.head()


# # Segregating the Dataframe for Analyzing Individual Conditions

# In[14]:


X_birth = X[(X['condition'] == 'Birth Control')]
X_dep = X[(X['condition'] == 'Depression')]
X_bp = X[(X['condition'] == 'High Blood Pressure')]
X_diab = X[(X['condition'] == 'Diabetes, Type 2')]


# In[15]:


X_birth.head()


# # Importing the WordCloud Library

# In[16]:


from wordcloud import WordCloud


# # WordCloud for Birth Control

# In[17]:


plt.figure(figsize = (20,20)) # Text that is Fake News Headlines
wc = WordCloud(max_words = 500 , width = 1600 , height = 800).generate(" ".join(X_birth.review))
plt.imshow(wc , interpolation = 'bilinear')
plt.title('Word cloud for Birth control',fontsize=14)


# # WordCloud for Depression

# In[18]:


plt.figure(figsize = (20,20)) # Text that is Fake News Headlines
wc = WordCloud(max_words = 500 , width = 1600 , height = 800).generate(" ".join(X_dep.review))
plt.imshow(wc , interpolation = 'bilinear')
plt.title('Word cloud for Depression',fontsize=14)


# # WordCloud for Blood Pressure

# In[19]:


plt.figure(figsize = (20,20)) # Text that is Fake News Headlines
wc = WordCloud(max_words = 500 , width = 1600 , height = 800).generate(" ".join(X_bp.review))
plt.imshow(wc , interpolation = 'bilinear')
plt.title('Word cloud for High Blood Pressure',fontsize=14)


# # WordCloud for Diabetes Type 2

# In[20]:


plt.figure(figsize = (20,20)) # Text that is Fake News Headlines
wc = WordCloud(max_words = 500 , width = 1600 , height = 800).generate(" ".join(X_diab.review))
plt.imshow(wc , interpolation = 'bilinear')
plt.title('Word cloud for Diabetes Type 2',fontsize=14)


# # Data Preprocessing

# In[29]:


X['review'][2]


# In[33]:


X['review'][11]


# # Removing Double Quotes from all the Reviews present in the Dataset

# In[21]:


for i, col in enumerate(X.columns):
    X.iloc[:, i] = X.iloc[:, i].str.replace('"', '')


# # Setting the Width of the Column to Maximum

# In[22]:


pd.set_option('max_colwidth', -1)


# In[39]:


X.head()


# # Removing the StopWords

# ## Importing the StopWords Library

# In[38]:


import nltk
nltk.download('stopwords')


# ## StopWords in the English Language

# In[40]:


from nltk.corpus import stopwords


# In[41]:


stop = stopwords.words('english')


# In[42]:


stop


# # Stemming

# ## Importing the PorterStemmer Library

# In[26]:


from nltk.stem import PorterStemmer


# In[27]:


porter = PorterStemmer()


# ## Trying out Stemming for Random Examples

# In[28]:


print(porter.stem("sportingly"))
print(porter.stem("very"))
print(porter.stem("troubled"))


# # Lemmatization

# ## Importing the WordNetLemmatizer Library

# In[29]:


from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')


# In[30]:


lemmatizer = WordNetLemmatizer()


# ## Trying out Lemmatization for Random Examples

# In[31]:


print(lemmatizer.lemmatize("sportingly"))
print(lemmatizer.lemmatize("very"))
print(lemmatizer.lemmatize("troubled"))


# # Converting the Reviews to Individual Important Words

# ## Importing the BeautifulSoup and Re Libraries

# In[32]:


from bs4 import BeautifulSoup
import re


# ## Creating a Function for the Required Conversion

# In[33]:


def review_to_words(raw_review):
    # 1. Delete HTML 
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. lower letters
    words = letters_only.lower().split()
    # 5. Stopwords 
    meaningful_words = [w for w in words if not w in stop]
    # 6. lemmitization
    lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    # 7. space join words
    return( ' '.join(lemmitize_words))


# In[43]:


X['review_clean'] = X['review'].apply(review_to_words)


# In[53]:


X.head()


# # Creating Features and Target Variables

# In[51]:


X_feat = X['review_clean']
y = X['condition']


# In[52]:


X_feat_2 = X['tokenized']


# # Performing the Train - Test Split

# In[54]:


X_train, X_test, y_train, y_test = train_test_split(X_feat, y, stratify = y, test_size = 0.2, random_state=0)


# In[50]:


X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_feat_2, y, stratify = y, test_size = 0.2, random_state=0)


# # Creating a Function for Plotting Confusion Matrix

# In[81]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# # Bag of Words Approach

# In[55]:


count_vectorizer = CountVectorizer(stop_words = 'english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)


# In[102]:


count_train


# ## Logistic Regression Model

# In[67]:


from sklearn.metrics import f1_score, accuracy_score, plot_confusion_matrix, auc, confusion_matrix, precision_score, recall_score


# In[56]:


from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(random_state = 2)
lr_model.fit(count_train, y_train)


# In[88]:


preds = lr_model.predict(count_test)
score = metrics.accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, pos_label='positive', average= 'macro')
recall = recall_score(y_test, preds, pos_label='positive', average='macro')
f1 = f1_score(y_test, preds, pos_label='positive', average='macro')

print("accuracy: ", score * 100)
print("precision: ", precision * 100)
print("recall: ", recall * 100)
print("f1_score: ", f1)

cm = metrics.confusion_matrix(y_test, preds, labels=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])
plot_confusion_matrix(cm, classes=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])


# ## Decision Tree Classifier

# In[60]:


from sklearn.tree import DecisionTreeClassifier


# In[89]:


dt_model = DecisionTreeClassifier(random_state = 2)
dt_model.fit(count_train, y_train)

preds = dt_model.predict(count_test)
score = metrics.accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, pos_label='positive', average= 'macro')
recall = recall_score(y_test, preds, pos_label='positive', average='macro')
f1 = f1_score(y_test, preds, pos_label='positive', average='macro')

print("accuracy: ", score * 100)
print("precision: ", precision * 100)
print("recall: ", recall * 100)
print("f1_score: ", f1)

cm = metrics.confusion_matrix(y_test, preds, labels=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])
plot_confusion_matrix(cm, classes=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])


# ## Random Forest Classifier

# In[63]:


from sklearn.ensemble import RandomForestClassifier


# In[95]:


random_model = RandomForestClassifier(max_depth = 2, random_state = 2)
random_model.fit(count_train, y_train)

preds = random_model.predict(count_test)
score = metrics.accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, pos_label='positive', average= 'macro')
recall = recall_score(y_test, preds, pos_label='positive', average='macro')
f1 = f1_score(y_test, preds, pos_label='positive', average='macro')

print("accuracy: ", score * 100)
print("precision: ", precision * 100)
print("recall: ", recall * 100)
print("f1_score: ", f1)

cm = metrics.confusion_matrix(y_test, preds, labels=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])
plot_confusion_matrix(cm, classes=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])


# ## Naive Bayes Machine Learning Model

# In[96]:


mnb = MultinomialNB()
mnb.fit(count_train, y_train)
pred = mnb.predict(count_test)

score = metrics.accuracy_score(y_test, pred)
precision = precision_score(y_test, preds, pos_label='positive', average= 'macro')
recall = recall_score(y_test, preds, pos_label='positive', average='macro')
f1 = f1_score(y_test, preds, pos_label='positive', average='macro')

print("accuracy: ", score * 100)
print("precision: ", precision * 100)
print("recall: ", recall * 100)
print("f1_score: ", f1)

cm = metrics.confusion_matrix(y_test, pred, labels=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])
plot_confusion_matrix(cm, classes=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])


# In[ ]:





# ## Passive Aggresive Classifier Machine Learning Model

# In[97]:


from sklearn.linear_model import PassiveAggressiveClassifier,LogisticRegression

passive = PassiveAggressiveClassifier()
passive.fit(count_train, y_train)
pred = passive.predict(count_test)

score = metrics.accuracy_score(y_test, pred)

precision = precision_score(y_test, preds, pos_label='positive', average= 'macro')
recall = recall_score(y_test, preds, pos_label='positive', average='macro')
f1 = f1_score(y_test, preds, pos_label='positive', average='macro')

print("accuracy: ", score * 100)
print("precision: ", precision * 100)
print("recall: ", recall * 100)
print("f1_score: ", f1)

cm = metrics.confusion_matrix(y_test, pred, labels=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])
plot_confusion_matrix(cm, classes=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])


# # TFIDF Approach

# In[99]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8)
tfidf_train_2 = tfidf_vectorizer.fit_transform(X_train)
tfidf_test_2 = tfidf_vectorizer.transform(X_test)


# ## Logistic Regression Model

# In[100]:


from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(random_state = 2)
lr_model.fit(tfidf_train_2, y_train)


# In[101]:


preds = lr_model.predict(tfidf_test_2)
score = metrics.accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, pos_label='positive', average= 'macro')
recall = recall_score(y_test, preds, pos_label='positive', average='macro')
f1 = f1_score(y_test, preds, pos_label='positive', average='macro')

print("accuracy: ", score * 100)
print("precision: ", precision * 100)
print("recall: ", recall * 100)
print("f1_score: ", f1)

cm = metrics.confusion_matrix(y_test, preds, labels=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])
plot_confusion_matrix(cm, classes=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])


# 

# ## Decision Tree Classifier

# In[102]:


from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state = 2)
dt_model.fit(tfidf_train_2, y_train)


# In[103]:


preds = dt_model.predict(tfidf_test_2)
score = metrics.accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, pos_label='positive', average= 'macro')
recall = recall_score(y_test, preds, pos_label='positive', average='macro')
f1 = f1_score(y_test, preds, pos_label='positive', average='macro')

print("accuracy: ", score * 100)
print("precision: ", precision * 100)
print("recall: ", recall * 100)
print("f1_score: ", f1)

cm = metrics.confusion_matrix(y_test, preds, labels=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])
plot_confusion_matrix(cm, classes=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])


# ## Random Forest Classifier

# In[104]:


from sklearn.ensemble import RandomForestClassifier

random_model = RandomForestClassifier(max_depth = 2, random_state = 2)
random_model.fit(tfidf_train_2, y_train)


# In[105]:


preds = random_model.predict(tfidf_test_2)
score = metrics.accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, pos_label='positive', average= 'macro')
recall = recall_score(y_test, preds, pos_label='positive', average='macro')
f1 = f1_score(y_test, preds, pos_label='positive', average='macro')

print("accuracy: ", score * 100)
print("precision: ", precision * 100)
print("recall: ", recall * 100)
print("f1_score: ", f1)

cm = metrics.confusion_matrix(y_test, preds, labels=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])
plot_confusion_matrix(cm, classes=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])


# ## Naive Bayes Machine Learning Model

# In[106]:


mnb_tf = MultinomialNB()
mnb_tf.fit(tfidf_train_2, y_train)
pred = mnb_tf.predict(tfidf_test_2)

score = metrics.accuracy_score(y_test, pred)
precision = precision_score(y_test, preds, pos_label='positive', average= 'macro')
recall = recall_score(y_test, preds, pos_label='positive', average='macro')
f1 = f1_score(y_test, preds, pos_label='positive', average='macro')

print("accuracy: ", score * 100)
print("precision: ", precision * 100)
print("recall: ", recall * 100)
print("f1_score: ", f1)

cm = metrics.confusion_matrix(y_test, pred, labels=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])
plot_confusion_matrix(cm, classes=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])


# ## Passive Aggressive Classifier Machine Learning Model

# In[107]:


passive = PassiveAggressiveClassifier()
passive.fit(tfidf_train_2, y_train)
pred = passive.predict(tfidf_test_2)

score = metrics.accuracy_score(y_test, pred)
precision = precision_score(y_test, preds, pos_label='positive', average= 'macro')
recall = recall_score(y_test, preds, pos_label='positive', average='macro')
f1 = f1_score(y_test, preds, pos_label='positive', average='macro')

print("accuracy: ", score * 100)
print("precision: ", precision * 100)
print("recall: ", recall * 100)
print("f1_score: ", f1)

cm = metrics.confusion_matrix(y_test, pred, labels=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])
plot_confusion_matrix(cm, classes=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])


# # TFIDF: Bigrams Approach

# In[108]:


tfidf_vectorizer2 = TfidfVectorizer(stop_words='english', max_df=0.8, ngram_range=(1,2))
tfidf_train_2 = tfidf_vectorizer2.fit_transform(X_train)
tfidf_test_2 = tfidf_vectorizer2.transform(X_test)


# ## Logistic Regression Model

# In[109]:


from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(random_state = 2)
lr_model.fit(tfidf_train_2, y_train)


# In[110]:


preds = lr_model.predict(tfidf_test_2)
score = metrics.accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, pos_label='positive', average= 'macro')
recall = recall_score(y_test, preds, pos_label='positive', average='macro')
f1 = f1_score(y_test, preds, pos_label='positive', average='macro')

print("accuracy: ", score * 100)
print("precision: ", precision * 100)
print("recall: ", recall * 100)
print("f1_score: ", f1)

cm = metrics.confusion_matrix(y_test, preds, labels=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])
plot_confusion_matrix(cm, classes=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])


# ## Decision Tree Classifier

# In[111]:


from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state = 2)
dt_model.fit(tfidf_train_2, y_train)


# In[112]:


preds = dt_model.predict(tfidf_test_2)
score = metrics.accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, pos_label='positive', average= 'macro')
recall = recall_score(y_test, preds, pos_label='positive', average='macro')
f1 = f1_score(y_test, preds, pos_label='positive', average='macro')

print("accuracy: ", score * 100)
print("precision: ", precision * 100)
print("recall: ", recall * 100)
print("f1_score: ", f1)

cm = metrics.confusion_matrix(y_test, preds, labels=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])
plot_confusion_matrix(cm, classes=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])


# ## Random Forest Classifier

# In[113]:


from sklearn.ensemble import RandomForestClassifier

random_model = RandomForestClassifier(max_depth = 2, random_state = 2)
random_model.fit(tfidf_train_2, y_train)


# In[114]:


preds = random_model.predict(tfidf_test_2)
score = metrics.accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, pos_label='positive', average= 'macro')
recall = recall_score(y_test, preds, pos_label='positive', average='macro')
f1 = f1_score(y_test, preds, pos_label='positive', average='macro')

print("accuracy: ", score * 100)
print("precision: ", precision * 100)
print("recall: ", recall * 100)
print("f1_score: ", f1)

cm = metrics.confusion_matrix(y_test, preds, labels=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])
plot_confusion_matrix(cm, classes=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])


# ## Passive Aggressive Classifier Machine Learning Model

# In[126]:


pass_tf = PassiveAggressiveClassifier()
pass_tf.fit(tfidf_train_2, y_train)
pred = pass_tf.predict(tfidf_test_2)

score = metrics.accuracy_score(y_test, pred)
precision = precision_score(y_test, preds, pos_label='positive', average= 'macro')
recall = recall_score(y_test, preds, pos_label='positive', average='macro')
f1 = f1_score(y_test, preds, pos_label='positive', average='macro')

print("accuracy: ", score * 100)
print("precision: ", precision * 100)
print("recall: ", recall * 100)
print("f1_score: ", f1)

cm = metrics.confusion_matrix(y_test, pred, labels=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])
plot_confusion_matrix(cm, classes=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])


# # TFIDF : Trigrams Approach

# In[117]:


tfidf_vectorizer3 = TfidfVectorizer(stop_words='english', max_df=0.8, ngram_range=(1,3))
tfidf_train_3 = tfidf_vectorizer3.fit_transform(X_train)
tfidf_test_3 = tfidf_vectorizer3.transform(X_test)


# ## Logistic Regression Model

# In[118]:


from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(random_state = 2)
lr_model.fit(tfidf_train_3, y_train)


# In[119]:


preds = lr_model.predict(tfidf_test_3)
score = metrics.accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, pos_label='positive', average= 'macro')
recall = recall_score(y_test, preds, pos_label='positive', average='macro')
f1 = f1_score(y_test, preds, pos_label='positive', average='macro')

print("accuracy: ", score * 100)
print("precision: ", precision * 100)
print("recall: ", recall * 100)
print("f1_score: ", f1)

cm = metrics.confusion_matrix(y_test, preds, labels=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])
plot_confusion_matrix(cm, classes=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])


# ## Decision Tree Classifier

# In[120]:


from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state = 2)
dt_model.fit(tfidf_train_3, y_train)


# In[121]:


preds = dt_model.predict(tfidf_test_3)
score = metrics.accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, pos_label='positive', average= 'macro')
recall = recall_score(y_test, preds, pos_label='positive', average='macro')
f1 = f1_score(y_test, preds, pos_label='positive', average='macro')

print("accuracy: ", score * 100)
print("precision: ", precision * 100)
print("recall: ", recall * 100)
print("f1_score: ", f1)

cm = metrics.confusion_matrix(y_test, preds, labels=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])
plot_confusion_matrix(cm, classes=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])


# ## Random Forest Classifier
# 

# In[122]:


from sklearn.ensemble import RandomForestClassifier

random_model = RandomForestClassifier(max_depth = 2, random_state = 2)
random_model.fit(tfidf_train_3, y_train)


# In[123]:


preds = random_model.predict(tfidf_test_3)
score = metrics.accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, pos_label='positive', average= 'macro')
recall = recall_score(y_test, preds, pos_label='positive', average='macro')
f1 = f1_score(y_test, preds, pos_label='positive', average='macro')

print("accuracy: ", score * 100)
print("precision: ", precision * 100)
print("recall: ", recall * 100)
print("f1_score: ", f1)

cm = metrics.confusion_matrix(y_test, preds, labels=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])
plot_confusion_matrix(cm, classes=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])


# ## Passive Aggresive Classifier Machine Learning Model

# In[125]:


pass_tf = PassiveAggressiveClassifier()
pass_tf.fit(tfidf_train_3, y_train)
pred = pass_tf.predict(tfidf_test_3)

score = metrics.accuracy_score(y_test, pred)
precision = precision_score(y_test, preds, pos_label='positive', average= 'macro')
recall = recall_score(y_test, preds, pos_label='positive', average='macro')
f1 = f1_score(y_test, preds, pos_label='positive', average='macro')

print("accuracy: ", score * 100)
print("precision: ", precision * 100)
print("recall: ", recall * 100)
print("f1_score: ", f1)

cm = metrics.confusion_matrix(y_test, pred, labels=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])
plot_confusion_matrix(cm, classes=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])


# # Most Important Features

# ## Creating a Function to find out the Most Important Features for a Class

# In[75]:


def most_informative_feature_for_class(vectorizer, classifier, classlabel, n=10):
    labelid = list(classifier.classes_).index(classlabel)
    feature_names = vectorizer.get_feature_names()
    topn = sorted(zip(classifier.coef_[labelid], feature_names))[-n:]

    for coef, feat in topn:
        print (classlabel, feat, coef)


# ## Most Informative Features for the Class 'Birth Control'

# In[76]:


most_informative_feature_for_class(tfidf_vectorizer, pass_tf, 'Birth Control')


# ## Most Informative Features for the Class 'Depression'

# In[77]:


most_informative_feature_for_class(tfidf_vectorizer, pass_tf, 'Depression')


# ## Most Informative Features for the Class 'High Blood Pressure'

# In[79]:


most_informative_feature_for_class(tfidf_vectorizer, pass_tf, 'High Blood Pressure')


# ## Most Informative Features for the Class 'Diabetes, Type 2'

# In[80]:


most_informative_feature_for_class(tfidf_vectorizer, pass_tf, 'Diabetes, Type 2')


# In[81]:


X.tail()


# # Sample Predictions

# In[88]:


tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

pass_tf = PassiveAggressiveClassifier()
pass_tf.fit(tfidf_train, y_train)
pred = pass_tf.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])
plot_confusion_matrix(cm, classes=['Birth Control', 'Depression','Diabetes, Type 2','High Blood Pressure'])


# In[89]:


text = ["I have only been on Tekturna for 9 days. The effect was immediate. I am also on a calcium channel blocker (Tiazac) and hydrochlorothiazide. I was put on Tekturna because of palpitations experienced with Diovan (ugly drug in my opinion, same company produces both however). The palpitations were pretty bad on Diovan, 24 hour monitor by EKG etc. After a few days of substituting Tekturna for Diovan, there are no more palpitations."]
test = tfidf_vectorizer.transform(text)
pred1 = pass_tf.predict(test)[0]
pred1


# In[90]:


text =["This is the third med I've tried for anxiety and mild depression. Been on it for a week and I hate it so much. I am so dizzy, I have major diarrhea and feel worse than I started. Contacting my doc in the am and changing asap."]
test = tfidf_vectorizer.transform(text)
pred1 = pass_tf.predict(test)[0]
pred1


# In[104]:


text = ["I just got diagnosed with type 2. My doctor prescribed Invokana and metformin from the beginning. My sugars went down to normal by the second week. I am losing so much weight. No side effects yet. Miracle medicine for me"]
test = tfidf_vectorizer.transform(text)
pred1 = pass_tf.predict(test)[0]
pred1


# In[105]:


print("Hello World!!")


# In[107]:


conda install gensim


# In[108]:


# conda install python-Levenshtein


# # Word2Vec Approach

# ## Importing the Gensim Library for Implementing Word2Vec

# In[109]:


import gensim


# In[113]:


# nltk.download('punkt');


# ## Creating a Function to Tokenize Pandas Dataframe Column

# In[114]:


def tokenize(column):
    """Tokenizes a Pandas dataframe column and returns a list of tokens.

    Args:
        column: Pandas dataframe column (i.e. df['text']).

    Returns:
        tokens (list): Tokenized list, i.e. [Donald, Trump, tweets]
    """

    tokens = nltk.word_tokenize(column)
    return [w for w in tokens if w.isalpha()]  


# ## Performing Tokenization

# In[119]:


X["tokensized"] = X.apply(lambda x: tokenize(x["review_clean"]), axis = 1)


# In[121]:


X.rename(columns = {"tokensized": "tokenized"}, inplace = True)


# In[122]:


X.head()


# In[127]:


X_train_2.head()


# In[129]:


model = gensim.models.Word2Vec(window = 10,
                              min_count = 2,
                              workers = 4)


# In[130]:


model.build_vocab(X_train_2, progress_per = 1000)


# In[131]:


model.epochs


# In[132]:


model.train(X_train_2, total_examples = model.corpus_count, epochs = model.epochs)


# In[133]:


model.save("/Users/akshitkamboj/Desktop/Word2VecModel.model")


# In[138]:


model.wv.most_similar("side")


# In[142]:


y_train


# 

# In[96]:


# conda install -c conda-forge transformers


# In[100]:


conda create -n simplet python=3.7 pandas tqdm
conda activate simplet


# In[101]:


from simpletransformers.ner import NERModel
from transformers import AutoTokenizer
import pandas as pd
import logging


# In[110]:


X.head()


# In[112]:


X_train.head()


# In[ ]:





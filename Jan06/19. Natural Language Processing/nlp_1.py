import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

dataset = pd.read_csv("Restaurant_Reviews.tsv", sep="\t")

#nltk.download('stopwords')
stops = stopwords.words('english')
ps = PorterStemmer()
corpus = []
for i in np.arange(0, dataset.shape[0]):
    review = dataset['Review'][i]
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stops)]
    review = ' '.join(review)
    corpus.append(review)

############### Count Vectorization ######################
vectorizer = CountVectorizer(max_features=800)
X = vectorizer.fit_transform(corpus).toarray()
print(vectorizer.get_feature_names_out())
y = dataset['Liked']

rf = RandomForestClassifier(random_state=2022)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
print(rf.get_params())
params = {'max_features':[10, 50, 100, 200]}

gcv = GridSearchCV(rf, param_grid=params, verbose = 3, scoring='roc_auc', cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

############### TF-IDF Vectorization #####################

tf_idf = TfidfVectorizer(max_features=800)
X = tf_idf.fit_transform(corpus).toarray()
print(tf_idf.get_feature_names_out())
y = dataset['Liked']


rf = RandomForestClassifier(random_state=2022)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
print(rf.get_params())
params = {'max_features':[10, 50, 100, 200]}

gcv = GridSearchCV(rf, param_grid=params, verbose = 3, scoring='roc_auc', cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

########################## SPAM ############################
import os
os.chdir(r"C:\Training\Kaggle\Datasets\SPAM")

spam = pd.read_csv("SPAM text message 20170820 - Data.csv")
y = spam['Category'].map({'spam':1, 'ham':0})

stops = stopwords.words('english')
ps = PorterStemmer()
corpus = []
for i in np.arange(0, spam.shape[0]):
    review = spam['Message'][i]
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stops)]
    review = ' '.join(review)
    corpus.append(review)

############### Count Vectorization ######################
vectorizer = CountVectorizer(max_features=800)
X = vectorizer.fit_transform(corpus).toarray()
print(vectorizer.get_feature_names_out())


rf = RandomForestClassifier(random_state=2022)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
print(rf.get_params())
params = {'max_features':[10, 50, 100, 200]}

gcv = GridSearchCV(rf, param_grid=params, verbose = 3, scoring='roc_auc', cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)


############### TF-IDF Vectorization #####################

tf_idf = TfidfVectorizer(max_features=800)
X = tf_idf.fit_transform(corpus).toarray()
print(tf_idf.get_feature_names_out())

rf = RandomForestClassifier(random_state=2022)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
print(rf.get_params())
params = {'max_features':[10, 50, 100, 200]}

gcv = GridSearchCV(rf, param_grid=params, verbose = 3, scoring='roc_auc', cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

####################### News Sarcasm #########################
import os
os.chdir(r"C:\Training\Kaggle\Datasets\News Sarcasm")

df = pd.read_json('Sarcasm_Headlines_Dataset_v2.json',lines=True)
df = df[['is_sarcastic', 'headline']]

ps = PorterStemmer()
corpus = []
for i in np.arange(0, df.shape[0]):
    review = df['headline'][i]
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stops)]
    review = ' '.join(review)
    corpus.append(review)



############### Count Vectorization ######################
vectorizer = CountVectorizer(max_features=800)
X = vectorizer.fit_transform(corpus).toarray()
print(vectorizer.get_feature_names_out())
y = df['is_sarcastic']

rf = RandomForestClassifier(random_state=2022)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
print(rf.get_params())
params = {'max_features':[10, 50, 100, 200]}

gcv = GridSearchCV(rf, param_grid=params, verbose = 3, scoring='roc_auc', cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)


############### TF-IDF Vectorization #####################

tf_idf = TfidfVectorizer(max_features=800)
X = tf_idf.fit_transform(corpus).toarray()
print(tf_idf.get_feature_names_out())

rf = RandomForestClassifier(random_state=2022)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
print(rf.get_params())
params = {'max_features':[10, 50, 100, 200]}

gcv = GridSearchCV(rf, param_grid=params, verbose = 3, scoring='roc_auc', cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
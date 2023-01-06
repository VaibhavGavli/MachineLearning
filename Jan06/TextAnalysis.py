import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer

dataset =pd.read_csv("Restaurant_Reviews.tsv",sep="\t")

#nltk.download('stopwords')

stops=stopwords.words('english')


ps=PorterStemmer()
corpus=[]

for i in np.arange(0,dataset.shape[0]):
    review=dataset['Review'][i]
    review=re.sub('[^a-zA-Z]',' ',review)
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in set(stops)]
    review=' '.join(review)
    corpus.append(review)
    
vectorizer = CountVectorizer(max_features=800)
X = vectorizer.fit_transform(corpus).toarray()
print(vectorizer.get_feature_names_out())

y=dataset['Liked']

rf=RandomForestClassifier(random_state=2022)
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
print(rf.get_params())

params={'max_features':[10,50,100,200]}

gcv=GridSearchCV(rf,param_grid=params,verbose=3,scoring='roc_auc',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


############ TF-IDF Vectorization ########
from sklearn.feature_extraction.text import TfidfVectorizer

dataset =pd.read_csv("Restaurant_Reviews.tsv",sep="\t")

#nltk.download('stopwords')

stops=stopwords.words('english')


ps=PorterStemmer()
corpus=[]

for i in np.arange(0,dataset.shape[0]):
    review=dataset['Review'][i]
    review=re.sub('[^a-zA-Z]',' ',review)
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in set(stops)]
    review=' '.join(review)
    corpus.append(review)
    
vectorizer = TfidfVectorizer(max_features=800)
X = vectorizer.fit_transform(corpus).toarray()
print(vectorizer.get_feature_names_out())

y=dataset['Liked']

rf=RandomForestClassifier(random_state=2022)
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
print(rf.get_params())

params={'max_features':[10,50,100,200]}

gcv=GridSearchCV(rf,param_grid=params,verbose=3,scoring='roc_auc',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)
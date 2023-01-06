import nltk
from nltk import PorterStemmer

PorterStemmer().stem('dance')

PorterStemmer().stem('dancer')

PorterStemmer().stem('dancing')


PorterStemmer().stem('computer')

PorterStemmer().stem('computing')

PorterStemmer().stem('computational')


PorterStemmer().stem('compute')

############## stopswords #################

nltk.download('stopwords')
from nltk.corpus import stopwords
stops=stopwords.words('english')
print(stops)
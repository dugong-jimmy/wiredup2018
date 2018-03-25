from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics import accuracy_score 
from sklearn.base import TransformerMixin 
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import pandas
import spacy
import string
punctuations = string.punctuation

parser = spacy.load('en_core_web_sm')

dataset = pandas.read_csv('all-reviews.csv', delimiter=',')

trainset = dataset.sample(1000)
testset = dataset.sample(50)
#dataset = pandas.read_csv('top-100.csv', delimiter=',')
#testset = pandas.read_csv('testing.csv', delimiter=',')

#Custom transformer using spaCy 
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}

# Basic utility function to clean the text 
def clean_text(text):     
    return text.strip().lower()

#Create spacy tokenizer that parses a sentence and generates tokens
#these can also be replaced by word vectors 
def spacy_tokenizer(sentence):
    tokens = parser(sentence)
    tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
    tokens = [tok for tok in tokens if (tok not in stopwords and tok not in punctuations)]
    return tokens

#create vectorizer object to generate feature vectors, we will use custom spacy’s tokenizer
vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1)) 
classifier = LinearSVC()

# Create the  pipeline to clean, tokenize, vectorize, and classify 
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', vectorizer),
                 ('classifier', classifier)])

# Load sample data
traintext=[]
testtext=[]
trainrating=[]
testrating=[]
training=[]
testing=[]

for index, row in trainset.iterrows():
    training.append([row['text'], row['rating']])

for index, row in testset.iterrows():
    testing.append([row['text'], row['rating']])
"""
train = [('I love this sandwich.', '3'),          
         ('this is an amazing place!', '4'),
         ('I feel very good about these beers.', '3'),
         ('this is my best work.', '5'),
         ("what an awesome view", '4'),
         ('I do not like this restaurant', '2'),
         ('I am tired of this stuff.', '1'),
         ("I can't deal with this", '1'),
         ('he is my sworn enemy!', '1'),          
         ('my boss is horrible.', '2')] 
test =   [('the beer was good.', '4'),     
         ('I do not enjoy my job', '2'),
         ("I ain't feelin dandy today.", '2'),
         ("I feel amazing!", '5'),
         ('Gary is a good friend of mine.', '3'),
         ("I can't believe I'm doing this.", '1')]
"""
# Create model and measure accuracy
pipe.fit([x[0] for x in training], [x[1] for x in training]) 
pred_data = pipe.predict([x[0] for x in testing]) 
for (sample, pred) in zip(testing, pred_data):
    print(sample, pred) 
print("Accuracy:", accuracy_score([x[1] for x in testing], pred_data))
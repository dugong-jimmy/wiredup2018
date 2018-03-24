import pandas
import spacy
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
nlp = spacy.load('en_core_web_sm')

# Load dataset
dataset = pandas.read_csv('top-100.csv', delimiter=',')

# shape
print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())
# class distribution
print(dataset.groupby('client').size())

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# histograms
dataset.hist()
plt.show()

# scatter plot matrix
scatter_matrix(dataset)
plt.show()

# scatter plot matrix
scatter_matrix(dataset)
plt.show()

for doc in nlp.pipe(iter(dataset['text'])):
    print(doc[0].text, doc[0].tag_)

#########################################################################################
##                   		                                                           ##  
##                   		                                                           ##  
##                            DATA SCIENCE FROM SCRATCH                                ##
##                   			                                                       ##  
##                   		                                                           ## 
##                   		                                                           ##  
##                   		                                                           ##  
##                   		                                                           ##  
##                   		                                                           ##  
##                   		                                                           ##  
##                   		                                                           ##   
#########################################################################################




# Everything after the hashtag in this line is a comment.
# This is to keep your sanity.

#########################################################################################
#########################     CHAPTER I: Python Quick Review    ######################### 
#########################################################################################

#First we import Numpy and Scipy.

import numpy as np
from scipy import stats

#Next we create a dataset by passing a list into Numpy array function.

dataset = np.array([3, 1, 4, 1, 1])

#We can easily calculate the mean , median and mode 

mean = np.mean(dataset)
print(mean)

median = np.median(dataset)
print('Median: {:.1f}'.format(median))

mode= stats.mode(dataset)
print(mode)
print('Mode: {}'.format(mode[0][0]))
print('{} appeared {} times in the dataset'.format(mode[0][0], mode[1][0]))


#Let us now see how covariance and correlation can be implemented in Python using Numpy and Scipy.

import numpy as np
x = np.random.normal(size=2)
y = np.random.normal(size=2)

#We stack x and y vertically to produce z using the line of code below.
z = np.vstack((x, y))

#The data in now in the correct form and we can pass it to Numpy covariance function.
c = np.cov(z.T)
print(c)

a = [1,4,6]
b = [1,2,3]

corr = pearsonr(a,b)
print(corr)


#########################################################################################
###################    Overview of Python Programming Language      ##################### 
#########################################################################################

#The code below outputs the string “Hello World!” 
print('Hello World!')

#The code below assigns an integer
a = 3
b = 4
c = a + b
print('The value of a is {}, while the value of b is {}, and their sum c is {}'.format(a, b, c))

a = 200
b = 33
if b > a:
  print("b is greater than a")
elif a == b:
  print("a and b are equal")
else:
  print("a is greater than b")
 
#The code below prints 0 and 1 to the screen.
for x in range(2):
  print(x)

#Python Function
def my_function(planet):
  print('Hello ' + planet)

my_function('Earth!')

# a single line comment using pound or hash symbol
'''
A multi-line
comment in Python
'''
print('Comments in Python!')

#Python Data Structures

my_list = ['apple', 'banana', 4, 20]
print(my_list)

#Lists can also be defined using the list constructor as shown below.

another_list = list(('a', 'b', 'c'))
print(another_list)

#Tuples are immutable, this means that we cannot change the values of a tuple, trying to do so would result in an error. Below is how tuples are created.
my_tuple = (1, 2, 3, 4)
print(my_tuple)
print(type(my_tuple))

#Using the inbuilt type function gives us the type of an object.
my_set = {1, 1, 2, 2, 2, 'three'}
print(my_set)
print(type(my_set))

#Dictionaries are created using curly braces 
#with each key pointing to its corresponding value.

my_dict = {'1': 'one', '2': 'two', '3': 'three'}
print(my_dict)
print(type(my_dict))




#########################################################################################
####################         Python Data Science Tools         ###################### 
#########################################################################################

#Numpy arrays can be initiated by nested Python lists. 
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])   # create a rank 2 array
print(type(a))
print(a.shape)

#Arrays can also be initialized randomly from a distribution such as the normal distribution. 
#Trainable parameters of a model such as the weights are usually initialized randomly.

b = np.random.random((2,2))  # create an array filled with random values
print(b)
print(b.shape)

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

# matrix product
print(np.dot(x, y))

#The code below shows how to create a Series object in Pandas.

import pandas as pd

s = pd.Series([1,3,5,np.nan,6,8])

print(s)

#To create a dataframe, we can run the following code.
df = pd.DataFrame(np.random.randn(6,4), columns=list('ABCD'))

print(df)

#Here is a simple usage of scipy that finds the inverse of a matrix.
from scipy import linalg
z = np.array([[1,2],[3,4]])

print(linalg.inv(z))

#Here is an example that uses Matplotlib to plot a sine waveform.

# magic command for Jupyter notebooks
%matplotlib inline

import matplotlib.pyplot as plt

# compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# plot the points using matplotlib
plt.plot(x, y)
plt.show()  # Show plot by calling plt.show()

 
#Scikit-Learn

# sample decision tree classifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

# load the iris datasets
dataset = datasets.load_iris()

# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(dataset.data, dataset.target)
print(model)

# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


#########################################################################################
##############                      K-Nearest Neighbors            ###################### 
######################################################################################### 

# The dataset can be downloaded from Kaggle https://www.kaggle.com/saurabh00007/iriscsv/downloads/Iris.csv/1.


#To begin let’s import all relevant libraries.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp

#Next we use Pandas to load the dataset which is contained in a CSV file 
dataset = pd.read_csv('Iris.csv')
dataset.head(5)

#In line with our observations, we separate the columns into features (X) and targets (y).

X = dataset.iloc[:, 1:5].values # select features ignoring non-informative column Id
y = dataset.iloc[:, 5].values # Species contains targets for our model

#To do this we leverage Scikit-Learn label encoder.

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y) # transform species names into categorical values

#Next we split our dataset into a training set and a test set 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#We can implement L2 distance in Python using Numpy as shown below.

def euclidean_distance(training_set, test_instance):
    # number of samples inside training set
    n_samples = training_set.shape[0]
    
    # create array for distances
    distances = np.empty(n_samples, dtype=np.float64)
    
    # euclidean distance calculation
    for i in range(n_samples):
        distances[i] = np.sqrt(np.sum(np.square(test_instance - training_set[i])))
        
    return distances

#Locating Neighbors

class MyKNeighborsClassifier():
    """
    Vanilla implementation of KNN algorithm.
    """
    
    def __init__(self, n_neighbors=5):
        self.n_neighbors=n_neighbors
        
    def fit(self, X, y):
        """
        Fit the model using X as array of features and y as array of labels.
        """
        n_samples = X.shape[0]
        # number of neighbors can't be larger then number of samples
        if self.n_neighbors > n_samples:
            raise ValueError("Number of neighbors can't be larger then number of samples in training set.")
        
        # X and y need to have the same number of samples
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y need to be equal.")
        
        # finding and saving all possible class labels
        self.classes_ = np.unique(y)
        
        self.X = X
        self.y = y
        
    def pred_from_neighbors(self, training_set, labels, test_instance, k):
        distances = euclidean_distance(training_set, test_instance)
        
        # combining arrays as columns
        distances = sp.c_[distances, labels]
        # sorting array by value of first column
        sorted_distances = distances[distances[:,0].argsort()]
        # selecting labels associeted with k smallest distances
        targets = sorted_distances[0:k,1]

        unique, counts = np.unique(targets, return_counts=True)
        return(unique[np.argmax(counts)])
        
        
    def predict(self, X_test):
        
        # number of predictions to make and number of features inside single sample
        n_predictions, n_features = X_test.shape
        
        # allocationg space for array of predictions
        predictions = np.empty(n_predictions, dtype=int)
        
        # loop over all observations
        for i in range(n_predictions):
            # calculation of single prediction
            predictions[i] = self.pred_from_neighbors(self.X, self.y, X_test[i, :], self.n_neighbors)

        return(predictions)


# instantiate learning model (k = 3)
my_classifier = MyKNeighborsClassifier(n_neighbors=3)

# fitting the model
my_classifier.fit(X_train, y_train)

# predicting the test set results
my_y_pred = my_classifier.predict(X_test)

#We then check the predicted classes against the ground truth labels 
#and use Scikit-Learn accuracy module to calculate the accuracy of our classifier.

from sklearn.metrics import confusion_matrix, accuracy_score
accuracy = accuracy_score(y_test, my_y_pred)*100
print('Accuracy: ' + str(round(accuracy, 2)) + ' %.')

 
#########################################################################################
##############                 CHAPTER 6: Regression               ###################### 
######################################################################################### 

#The dataset can be downloaded at https://www.kaggle.com/uciml/sms-spam-collection-dataset/downloads/spam.csv/1


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# comment the magic command below if not running in Jupyter notebook

#Next we load the dataset using Pandas and display the first 5 rows.
data = pd.read_csv('spam.csv', encoding='latin-1')
data.head(5)

#Let us plot a bar chart to visualize the distribution of legitimate and spam messages.

count_class = pd.value_counts(data['v1'], sort= True)
count_class.plot(kind='bar', color=[['blue', 'red']])
plt.title('Bar chart')
plt.show()

 
#We have to vectorize them to create new features. 
from sklearn.feature_extraction.text import CountVectorizer

f = CountVectorizer(stop_words = 'english')
X = f.fit_transform(data["v2"])
print(np.shape(X))


#Next we map our target variables into categories and split the dataset into train and test sets.
from sklearn.model_selection import train_test_split

data["v1"]=data["v1"].map({'spam':1,'ham':0})
X_train, X_test, y_train, y_test = train_test_split(X, data['v1'], test_size=0.25, random_state=42)

#The next step involves initializing the Naive Bayes model and training it on the data.
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(X_train, y_train)

#Finally, we gauge the model performance on the test set.

score = clf.score(X_test, y_test)
print('Accuracy: {}'.format(score))

 
#########################################################################################
##############                        Regression                   ###################### 
######################################################################################### 
 
#The dataset can be downloaded from this URL https://forge.scilab.org/index.php/p/rdataset/source/file/master/csv/MASS/Boston.csv
 
#First we import relevant libraries and load the dataset using Pandas.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# matplotlib magic command for Jupyter notebook
%matplotlib inline

dataset = pd.read_csv('Boston.csv')
dataset.head()


#Let us plot the relationship between one of the predictors and the price of a house 
plt.scatter(dataset['crim'], dataset['medv'])
plt.xlabel('Per capita crime rate by town')
plt.ylabel('Price')
plt.title("Prices vs Crime rate")

#Next we split our dataset into predictors and targets. Then we create a training and test set.

X = dataset.drop(['Unnamed: 0', 'medv'], axis=1)
y = dataset['medv']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# import linear regression classifier, initialize and fit the model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

#Having fit the classifier, we can use it to predict house prices using features in the test set.

y_pred = regressor.predict(x_test)

#The next step is to evaluate the classifier using metrics such as the mean square error and the coefficient of determination  R square.

from sklearn.metrics import mean_squared_error, r2_score

# The coefficients
print('Coefficients: \n', regressor.coef_)
# The mean squared error
print('Mean squared error: {:.2f}'.format(mean_squared_error(y_test, y_pred)))
# Explained variance score: 1 is perfect prediction
print('Variance score: {:.2f}'.format(r2_score(y_test, y_pred)))


#Finally, we can plot the predicted prices from the model against the ground truth (actual prices).

plt.scatter(y_test, y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")

 

#The scatter plot above shows a positive relationship between the predicted prices and actual prices. 
#Here is the code in its entirety.

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# load dataset
dataset = pd.read_csv('Boston.csv')
dataset.head()

# plot crime vs price
plt.scatter(dataset['crim'], dataset['medv'])
plt.xlabel('Per capita crime rate by town')
plt.ylabel('Price')
plt.title("Prices vs Crime rate")

# separate predictors and targets
X = dataset.drop(['Unnamed: 0', 'medv'], axis=1)
y = dataset['medv']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# import linear regression classifier, initialize and fit the model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score

# The coefficients
print('Coefficients: \n', regressor.coef_)
# The mean squared error
print('Mean squared error: {:.2f}'.format(mean_squared_error(y_test, y_pred)))
# Explained variance score: 1 is perfect prediction
print('Variance score: {:.2f}'.format(r2_score(y_test, y_pred)))

# plot predicted prices vs actual prices
plt.scatter(y_test, y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")

#Logistic Regression

#The dataset can be downloaded at: https://www.kaggle.com/uciml/pima-indians-diabetes-database/data

#Let us import relevant libraries and load the dataset to have a sense of what it contains.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('diabetes.csv')
dataset.head(5)

#Next we separate the columns in the dataset into features and labels. 
features = dataset.drop(['Outcome'], axis=1)
labels = dataset['Outcome']

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25)

# Training the model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

classifier.fit(features_train, labels_train)

The trained model can now be evaluated on the test set.

pred = classifier.predict(features_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print('Accuracy: {:.2f}'.format(accuracy))

#Generalized Linear Models
 
#First we import the Statsmodels package as shown below.
import statsmodels.api as sm

#Next we load the dataset and extract the explanatory variable (X).
data = sm.datasets.scotland.load()
# data.exog is the independent variable X
data.exog = sm.add_constant(data.exog)

# we import the appropriate model and instantiate an object from it. 
# Instantiate a poisson family model with the default link function.
poisson_model = sm.GLM(data.endog, data.exog, family=sm.families.Poisson())

#We then fit the model on the data.
poisson_results = poisson_model.fit()

#We can now print a summary of results to better understand the trained model.
print(poisson_results.summary())



#########################################################################################
##############            Decision Trees and Random Forest         ###################### 
######################################################################################### 

#It can be downloaded at https://gist.github.com/tijptjik/9408623/archive/b237fa5848349a14a14e5d4107dc7897c21951f5.zip

# First, lets load the dataset and use Pandas head method to have a look at it.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# comment the magic command below if not running in Jupyter notebook
%matplotlib inline

dataset = pd.read_csv('wine.csv')
dataset.head(5)

#The next thing we do is split the dataset into predictors and targets, sometimes referred to as features and labels respectively.
features = dataset.drop(['Wine'], axis=1)
labels = dataset['Wine']

#we divide the dataset into a train and test split.
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25)

#All that is left is for us to import the decision tree classifier and fit it to our data.
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()

classifier.fit(features_train, labels_train)

#We can now evaluate the trained model on the test set and print out the accuracy.

pred = classifier.predict(features_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print('Accuracy: {:.2f}'.format(accuracy))

#Random Forests

import numpy as np
import pandas as pd

# load dataset
dataset = pd.read_csv('wine.csv')

# separate features and labels
features = dataset.drop(['Wine'], axis=1)
labels = dataset['Wine']

# split dataset into train and test sets
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25)

# import random forest classifier from sklearn
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()

# fit classifier on data
classifier.fit(features_train, labels_train)

# predict classes of test set samples
pred = classifier.predict(features_test)

# evaluate classifier performance using accuracy metric
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print('Accuracy: {:.2f}'.format(accuracy))



#########################################################################################
##############                   Neural Networks                   ###################### 
######################################################################################### 

#As before, let’s import the necessary library/libraries so that we can work on the data:
import pandas as pd
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# reference: https://pypi.org/project/apyori/#description

from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
results = list(rules)
results_list = []
for i in range(0, len(results)):
    results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]))
print (results_list)

#########################################################################################
##############            CHAPTER 9: Reinforcement Learning        ###################### 
######################################################################################### 

#import the necessary libraries so that we can work on our data (and also for data visualization)
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline  #so plots can show in our Jupyter Notebook
#We then import the dataset and take a peek
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
dataset.head(10)
 
import random
N = 10000
d = 10
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward

#When we run and the code and visualize:	
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
 
#Reference: https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf


#########################################################################################
##############         CHAPTER 10: Artificial Neural Networks       ###################### 
######################################################################################### 

#if you want to get an idea how an ANN might look like in Python, here’s a sample code:
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1
for j in xrange(60000):
    l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
    l2_delta = (y - l2)*(l2*(1-l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)
#From https://iamtrask.github.io/2015/07/12/basic-python-network/

#########################################################################################
##############        CHAPTER 11: Natural Language Processing      ###################### 
######################################################################################### 


import pandas as pd
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
The next step is to import the necessary libraries and then clean the text:
import re    #importing Regular Expressions
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
We then create an empty list (corpus) where the meaningful words will be stored. And then run a for loop on the reviews and words. The goal here is to eliminate the “stopwords” (e.g. the, a, an, in). These words are often eliminated because they don’t add much meaning to statements.
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
If we view the corpus (without the stopwords), we’ll see these:
['wow love place',
 'crust good',
 'tasti textur nasti',
 'stop late may bank holiday rick steve recommend love',
 'select menu great price',
 'get angri want damn pho',
 'honeslti tast fresh',
 'potato like rubber could tell made ahead time kept warmer',
 'fri great',
 'great touch',
 'servic prompt',
 'would go back',
 'cashier care ever say still end wayyy overpr',
 'tri cape cod ravoli chicken cranberri mmmm',
 'disgust pretti sure human hair',
 'shock sign indic cash',

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



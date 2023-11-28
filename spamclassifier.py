"""
Naive Bayes algorithm is applied to create a model that can classify SMS messages as spam or not spam. 
The term 'Naive' in Naive Bayes comes from the fact that the algorithm considers the features that it is
using to make the predictions to be independent of each other, which may not always be the case

This is a binary classification problem as messages are classified as either 'Spam' or 'Not Spam' and nothing else. 
Also, this is a supervised learning problem, I am feeding a labelled dataset into the model, that it can learn from, to make future predictions.

Dataset originally compiled and posted on the UCI Machine Learning repository (column 1 - spam or ham label; second column - SMS message)

BoW is used to format the categorical data.The basic idea of BoW is to take a piece of text and count the frequency of the words in that text. 
It is important to note that the BoW concept treats each word individually and the order in which the words occur does not matter.
We organize the data into a matrix, with each document being a row and each word (token) being the column, 
and the corresponding (row, column) values being the frequency of occurrence of each word or token in that document.

Bayes Rule 
* P(A|B) = P(B|A)P(A)/P(B) - Posterior = Likelihood * Prior / Evidence - P(Condition|Test) = P(Test|Condition)P(Condition) / P(Test)
* priors - probabilities we are aware of related to the event in question , P(A)
* posteriors - probabilities we are looking to compute using the priors, sum of posteriors = 1
* Sensitivity or True Positive Rate - P(B|A) the probability of getting a positive test result given the condition is present, 
                                      This is a critical factor because it informs us about how well the test performs in identifying true positives P(Test|Condition)

Naive Bayes
* P(y|x1, ..., xn) = P(y)P(x1, ..., xn| y)/ P(x1, ...,xn) 
* Numerator P(y)P(x1, ..., xn|y) = P(y)P(x1|y)...P(xn|y)
* xn = feature vectors (indiviadual words)
* y = class variable  (candidate)
* theorem assumes that each of the feature vectors are independent of eachohter

sklearn Naive Bayes
multinomial Naive Bayes algorithm. This particular classifier is suitable for classification with discrete features (such as in our case, word counts for text classification). 
It takes in integer word counts as its input. On the other hand, Gaussian Naive Bayes is better suited for continuous data as it assumes that the input data 
has a Gaussian (normal) distribution.

Conclusion
One of the major advantages that Naive Bayes has over other classification algorithms is its ability to handle an extremely large number of features.
In our case, each word is treated as a feature and there are thousands of different words. Also, it performs well even with the presence of irrelevant 
features and is relatively unaffected by them. The other major advantage it has is its relative simplicity. Naive Bayes' works well right out of the
box and tuning its parameters is rarely ever necessary, except usually in cases where the distribution of the data is known. It rarely ever overfits the data. 
Another important advantage is that its model training and prediction times are very fast for the amount of data it can handle. 
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def bagOfWords_test():
    # bag of words (BoW)
    documents = ['Hello, how are you!',
                    'Win money, win from home.',
                    'Call me now.',
                    'Hello, Call hello you tomorrow?']
    count_vector = CountVectorizer()    #  text feature extraction method
    # print(vars(count_vector))
    count_vector.fit(documents)         # fit data to CountVectorizer object
    
    # create a matrix row = document number; column = feature name (a word)
    doc_array = count_vector.transform(documents).toarray()
    column_names = count_vector.get_feature_names_out() # list of words which have been categorized as features
    frequency_matrix = pd.DataFrame(doc_array, columns=column_names)


# load dataset
df = pd.read_table("./SMSSpamCollection", names=['label', 'sms_message'], delimiter='\t')
# print(df.head(5), '\n')


# data preprocessing 
df['label'] = df.label.map({'ham':0, 'spam':1})    #Convert the values in the 'label' column to numerical values using map method as follows: {'ham':0, 'spam':1} 


# split data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split( df["sms_message"], df["label"], random_state=1)    # x = sms_message, y = label 
print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {} \n'.format(X_test.shape[0]))


# apply bag of words processing to our daataset
count_vector = CountVectorizer()                             # Instantiate the CountVectorizer method
training_data = count_vector.fit_transform(X_train)          # Fit the training data and then return the matrix
testing_data = count_vector.transform(X_test)                # Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()

# bayes theorem - multinomial Naive Bayes algorithm
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)              # fit the training data into the classifier
predictions = naive_bayes.predict(testing_data)

# evaluate model 
"""
accuracy = correct predictions / total number of predictions
precision = True positives / (True positives + False positives)
Recall (sensitivity) = [True Positives/(True Positives + False Negatives)]
F1 score:
the weighted average of the precision and recall scores. This score can range from 0 to 1, with 1 being the best possible F1 score.
    * Precision tells us what proportion of messages we classified as spam, actually were spam. 
    * Recall (sensitivity) tells us what proportion of messages that actually were spam were classified by us as spam
"""
print('Accuracy score: ', format(accuracy_score(predictions, y_test)))
print('Precision score: ', format(precision_score(predictions, y_test)))
print('Recall score: ', format(recall_score(predictions, y_test)))
print('F1 score: ', format(f1_score(predictions, y_test)))
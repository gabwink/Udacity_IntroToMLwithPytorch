"""
Naive Bayes algorithm is applied to create a model that can classify SMS messages as spam or not spam.

This is a binary classification problem as messages are classified as either 'Spam' or 'Not Spam' and nothing else. 
Also, this is a supervised learning problem, I am feeding a labelled dataset into the model, that it can learn from, to make future predictions.

Dataset originally compiled and posted on the UCI Machine Learning repository (column 1 - spam or ham label; second column - SMS message)

BoW is used to format the categorical data.The basic idea of BoW is to take a piece of text and count the frequency of the words in that text. 
It is important to note that the BoW concept treats each word individually and the order in which the words occur does not matter.
We organize the data into a matrix, with each document being a row and each word (token) being the column, 
and the corresponding (row, column) values being the frequency of occurrence of each word or token in that document.
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

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
print(df.head(5), '\n')

# data preprocessing 
df['label'] = df.label.map({'ham':0, 'spam':1})    #Convert the values in the 'label' column to numerical values using map method as follows: {'ham':0, 'spam':1} 


# split data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split( df["sms_message"], df["label"], random_state=1)    # x = sms_message, y = label 
print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))


# apply bag of words processing to our daataset
count_vector = CountVectorizer()   # Instantiate the CountVectorizer method
count_vector.fit(X_train)          # Fit the training data and then return the matrix
doc_array = count_vector.transform(X_test) # Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()

# bayes theorem implementation from scratch 


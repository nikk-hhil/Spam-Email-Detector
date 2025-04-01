##importing the dependencies

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


##loading data from csv file to panadas dataframe
raw_mail_data = pd.read_csv('mail_data.csv')

##replacing numm values with null string

mail_data = raw_mail_data.where(pd.notnull(raw_mail_data),"")

##labeling spam mail as 0 and ham mail as 1
mail_data.loc[mail_data['Category'] == "spam", 'Category',] = 0
mail_data.loc[mail_data['Category'] == "ham", 'Category',] = 1

##seprating data as texts and label (message and [0 or 1])
X = mail_data['Message']
Y = mail_data['Category']

##splitting the data as train data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

##transforming the text data to feature vectors that can be used as input to the SVM model using TfidfVectorizer

feature_extraction = TfidfVectorizer(min_df = 1, stop_words = "english", lowercase = True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

##converting Y_train and Y_test values as integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')



##Training the SVM model with training data

model = LogisticRegression()
model.fit(X_train_features, Y_train) ##Training the Logistic Regression with training data

##Evaluating the trained model

prediction_on_training_data = model.predict(X_train_features)##prediction on training data
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)##accuracy on training data

##Evaluating the test model

prediction_on_test_data = model.predict(X_test_features)##prediction on training data
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)##accuracy on training data

##BUILDING A PREDICTIVE SYSTEM

##BUILDING A PREDICTIVE SYSTEM FOR ALL EMAILS

# Get all messages from the dataset
all_emails = mail_data['Message']  # Use the messages from your dataset

# Transform all emails to features
all_email_features = feature_extraction.transform(all_emails)

# Make predictions on all emails
predictions = model.predict(all_email_features)

# Print results for each email
print("\nPrediction Results for All Emails:")
for i, email in enumerate(all_emails):
    if (predictions[i] == 1):
        result = "HAM MAIL"
    else:
        result = "SPAM MAIL"
    
    # Print just a preview of each email (first 50 characters)
    email_preview = email[:50] + "..." if len(email) > 50 else email
    print(f"Email {i+1}: {email_preview} -> {result}")
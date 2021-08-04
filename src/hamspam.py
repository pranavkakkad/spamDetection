import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from preprocess_text import return_cleaned_text
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from DataFrameCreation import csv_generate

spam_df = csv_generate()
spam_df.to_csv("../data/spam_csv.csv")
# print(spam_df.head())

encoder = LabelEncoder()
spam_df["label"] = encoder.fit_transform(spam_df["label"])
# print(spam_df.head())
spam_df["emails"] = spam_df["emails"].apply(lambda x: x.lower())

spam_df, corpus = return_cleaned_text(spam_df)
corpus = spam_df["clean_email"].to_numpy()
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus)
y = spam_df["label"]

X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.2,random_state=42)

model = MultinomialNB()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy of the model is: ", accuracy)

cm = confusion_matrix(y_test,y_pred)

print("Confusion Matrix:")
print(cm)




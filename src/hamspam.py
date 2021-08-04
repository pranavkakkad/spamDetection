import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import sklearn
import os
from sklearn.preprocessing import LabelEncoder
from DataFrameCreation import csv_generate



spam_df = csv_generate()
spam_df.to_csv("../data/spam_csv.csv")
# print(spam_df.head())

encoder = LabelEncoder()
spam_df["label"] = encoder.fit_transform(spam_df["label"])
# print(spam_df.head())

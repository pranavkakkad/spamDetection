import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer

train_df = pd.read_csv("../data/spam.csv")
# print(train_df.head())
train_df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
train_df.columns = ["label","message"]
# print(train_df.head())

ps = PorterStemmer()
stop_words = stopwords.words("english")


sns.countplot(x="label",data=train_df)
# plt.show()

def text_clean(text):
    text = re.sub('^a-zA-Z',' ',text)
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    text = ' '.join(text).replace('  ',' ')
    return text

train_df["clean_text"] = train_df.message.apply(lambda x: text_clean(x))
# print(train_df.head())
cv = CountVectorizer(max_features=2500)

train = cv.fit_transform(train_df.clean_text).toarray()
# print(train.shape)
test = pd.get_dummies(train_df.label, drop_first=True)
test = np.array(test)


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score,precision_recall_curve,plot_precision_recall_curve


X_train, X_test,y_train,y_test = train_test_split(train, test,test_size=0.2, shuffle=False)

gaussian_classifier = GaussianNB()
multinomial_classifier = MultinomialNB()

gaussian_classifier.fit(X_train,y_train)
multinomial_classifier.fit(X_train,y_train)

gauss_pred = gaussian_classifier.predict(X_test)
mult_pred = multinomial_classifier.predict(X_test)

score_gaus = accuracy_score(gauss_pred,y_test)
score_mul = accuracy_score(mult_pred,y_test)

print("Gaussian accuracy %d" ,score_gaus)
print("Multinomial score %d" ,score_mul)

average_precision_gauss = average_precision_score(y_test, gauss_pred)
average_precision_multi = average_precision_score(y_test, mult_pred)

print(f"Avg Precision for GaussianNB: {average_precision_gauss:.2f}")
print(f"Avg Precision for MultinomialNB: {average_precision_multi:.2f}")

fig, ax = plt.subplots(figsize=(12,8))
curve = plot_precision_recall_curve(multinomial_classifier, X_test, y_test, ax=ax)
curve.ax_.set_title("MultinomialNB Precision/Recall Curve");
plt.show()

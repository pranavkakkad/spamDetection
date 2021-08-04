import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()
stop_words = stopwords.words("english")
corpus = []
def text_clean(text):
    text = re.sub('^a-zA-Z', ' ', text)
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    text = ' '.join(text).replace('  ', ' ')
    corpus.append(" ".join(text))
    return text


def return_cleaned_text(spam_df):
    spam_df["clean_email"] = spam_df["emails"].apply(lambda x: text_clean(x))

    return (spam_df,corpus)

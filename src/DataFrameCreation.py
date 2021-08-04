import os
import pandas as pd

parent_dir = "../data/hamnspam/"
spam_dir = os.listdir(parent_dir+"spam")
non_spam_dir = os.listdir(parent_dir+"ham")

def csv_generate():
    text = []
    label = []
    for i in ["ham/","spam/"]:
        files = os.listdir(parent_dir+i)
        for file in files:
            f = open((parent_dir+i+file),'r',encoding='latin-1')
            text.append(f.read())
            label.append(i)
    spam_df = pd.DataFrame({'emails':text,'label':label})
    return spam_df
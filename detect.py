# EDA Packages
import pandas as pd
import numpy as np
import random


# Machine Learning Packages
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
urls_data = pd.read_csv("urldata.csv")
y = urls_data["label"]
url_list = urls_data["url"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(url_list)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log_reg=LogisticRegression()
log_reg=log_reg.fit(x_train,y_train)
pickle.dump(log_reg,open('log_model.pkl','wb'))

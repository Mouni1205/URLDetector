# EDA Packages
import pandas as pd
import numpy as np
import random

# Machine Learning Packages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load Dataset
urls_data = pd.read_csv("urldata.csv")

# Extract Features and Labels
y = urls_data["label"]  # Labels
url_list = urls_data["url"]  # URL features

# Vectorize the URLs using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(url_list)

# Split Dataset into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
log_reg = LogisticRegression()
log_reg = log_reg.fit(X_train, y_train)

# Evaluate the Model
y_pred = log_reg.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the Model and Vectorizer
pickle.dump(log_reg, open('log_model.pkl', 'wb'))
pickle.dump(vectorizer, open('tfidf_vectorizer.pkl', 'wb'))

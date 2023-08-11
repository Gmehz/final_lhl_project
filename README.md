# final_lhl_project

## Project description
This project is focused on predicting diseases from symptoms using unsupervised machine learning and natural language processing (NLP) techniques. I am utilizing the Kaggle 'Symptom2Disease' dataset which includes various symptoms as text inputs and diseases as labels.

The goal is to accurately predict the disease given a set of symptoms. I approach this problem by using two machine learning models, Linear Support Vector Classifier (LinearSVC) and Random Forest Classifier, and three different vectorization techniques, namely Count Vectorizer, TF-IDF (Term Frequency-Inverse Document Frequency), and Word2Vec.

# As of now, the results indicate that the combination of the Random Forest Classifier and Count Vectorizer provides the best predictions.

Technologies
The project is implemented in Python, using several libraries for machine learning and natural language processing:

Scikit-learn: For implementing the Linear SVC and Random Forest Classifier models, as well as Count Vectorizer and TF-IDF.
Gensim: For Word2Vec implementation.
Pandas: For data manipulation and analysis.

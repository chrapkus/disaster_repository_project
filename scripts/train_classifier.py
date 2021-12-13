import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
import numpy as np
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import multioutput
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle

def load_data(database_path):
    """
    input: 
        database_path - path to sql database
    output:
        X_train - dataframe with messages used to trian model
        X_test - dataframe with messages used to test model
        Y_train - dataframe with features columns used to trian model
        Y_test - dataframe with features columns used to test model
        col_names - list of Y columns names 
    """

    # Connecting to database
    engine = create_engine(f'sqlite:///{database_path}')
    df = pd.read_sql("SELECT * FROM disaster", engine)

    # Sellecitng columns
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    col_names = list(Y.columns)

    return X_train, X_test, Y_train, Y_test, col_names


def tokenize(text):
    """
    input: 
        text - message that should be prepered and tokenize
    output:
        text - tocenize, cleaned, normalize, lemmatizated message
    """
    # Lower all text
    text = text.lower()

    # Normalize text - removing punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # Tokenize -> slit into list
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words_list = stopwords.words('english')
    tokens = [token.strip() for token in tokens if token not in stop_words_list]
    
    # Lemmatization
    lemmed = [WordNetLemmatizer().lemmatize(token, pos = 'v') for token in tokens]
    
    # Steaming
    stemmed = [PorterStemmer().stem(word) for word in lemmed]

    return stemmed


def build_model():
    """
    output:
        model with Grid Search for calssification
    """
        #setting pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', multioutput.MultiOutputClassifier (RandomForestClassifier()))
        ])

    # fbeta_score scoring object using make_scorer()
    scorer = 'f1_weighted'

    #model parameters for GridSearchCV
    parameters = {  'vect__max_df': (0.3, 0.5),
                    'clf__estimator__n_estimators': [10, 5],
                    'clf__estimator__min_samples_split': [2, 10]
              }
    cv = GridSearchCV(pipeline, param_grid= parameters, scoring = scorer, verbose =7, cv = 2)

    return cv

def evaluation(Y_test, Y_pred, col_names):
    """
    input:
        Y_test - data with expected output of model
        Y_pred - data with predicted data
        col_names - names of columns used for categorization
    output:
        dataframe with F1, Precision na Recall score for each category
    """

    metrics = []
    
    # Evaluate metrics for each set of labels
    for i in range(len(col_names)):
        precision = precision_score(Y_test[:, i], Y_pred[:, i])
        recall = recall_score(Y_test[:, i], Y_pred[:, i])
        f1 = f1_score(Y_test[:, i], Y_pred[:, i])
        
        metrics.append([f1, precision, recall])
    
    # store metrics
    metrics = np.array(metrics)
    data_metrics = pd.DataFrame(data = metrics, index = col_names, columns = ['F1', 'Precision', 'Recall'])
      
    return data_metrics 

def save_model(model, model_path):
    """ Saving model's best_estimator_ using pickle
    """
    pickle.dump(model.best_estimator_, open(model_path, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_path, model_path = sys.argv[1:]

    else:
        database_path = '..\\disaster.db'
        model_path = '..\\disaster_model.sav'

    print('Loading data...\n')
    X_train, X_test, Y_train, Y_test ,col_names = load_data(database_path)

    print('Creating moel...\n')
    model = build_model()

    print('Fitting model...\n')
    model.fit(X_train, Y_train)

    print('Predicting...\n')
    Y_pred = model.predict(X_test)

    print('Model evaluation...\n')
    results = evaluation(np.array(Y_test), Y_pred, col_names)
    print(results)

    print('Seving best model')
    save_model(model, model_path)

if __name__ == '__main__':
    main()
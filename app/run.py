import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../disaster.db')
df = pd.read_sql_table('disaster', engine)

# load model
model1 = joblib.load("../disaster_model.sav")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    zipped_lists = zip(genre_counts, genre_names)
    sorted_zipped_lists = sorted(zipped_lists, reverse=True)

    genre_names = [element for _, element in sorted_zipped_lists]
    genre_counts = [_ for _, element in sorted_zipped_lists]

    # Data for secound visualization
    categories_columns = df.columns[4:]
    category_count = df[categories_columns].sum().values.tolist()

    zipped_lists = zip(category_count, categories_columns)
    sorted_zipped_lists = sorted(zipped_lists, reverse=True)

    sorted_categories_columns = [element for _, element in sorted_zipped_lists]
    sorted_category_count = [_ for _, element in sorted_zipped_lists]

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
                {
            'data': [
                Bar(
                    x=sorted_categories_columns,
                    y=sorted_category_count
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }

    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model1.predict([query])[0]

    classification_results = dict(zip(df.columns[4:], classification_labels))


    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
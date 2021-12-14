# disaster_repository_project
Messages classification into 36 categeries.
Project consist of 3 parts:
*

## Installation neccecery libraries

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all libraries.

```bash
pip install pandas
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install re
pip install sqlalchemy
pip install pickle
pip install nltk
pip install flask
```

## Project structure
* app - folder with all files related to web app
  * templates - folder with html files
  * run.py - backend script that uses trained model to predict category of given message 
* data -  folder contains all the data used in project
  * categories.csv - data set with catgories of message
  * messages.csv - data set with messages
* scripts - folder with python scripts
  * process_data.py - script that cleans, merges, and deduplicates input data
  * train_classifier.py - scipt that train model to predict message categery

## Instructions:
All scripts uses relative file paths.
Before running each script navigate to its directory.
### Example
```bash
cd app
python run.py
```

* ETL pipeline that cleans data and stores it in database
    * `python process_data.py ../data/messages.csv ../data/categories.csv ../disaster.db`
    * If you will not pass arguments default values will be used
     `python process_data.py`
    
* ML pipeline that trains and saves classifier
    * `python train_classifier.py ../disaster.db ../disaster_model.sav`
    * If you will not pass arguments default values will be used
     `python train_classifier.py`
* Run web app with plotly visualizations
     * `python run.py`
* To see working web app open browser and go to http://0.0.0.0:3001/ or to http://localhost:3001/

## Licensing, Authors, Acknowledgements
This app was completed as part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).




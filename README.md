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




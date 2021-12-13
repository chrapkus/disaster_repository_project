import sys
import pandas as pd
from sqlalchemy import create_engine
import copy


def split_categories_columns(df):

    row = df['categories'][0]
    categories_column_names = row.split(';')
    categories_column_names= [category.split('-')[0] for category in categories_column_names]
    rename_dict = {}

    for index, category in enumerate(categories_column_names):
        rename_dict[index] = category
        
    df = df.join(df['categories'].str.split(';', expand=True).rename(columns = rename_dict))
    df = df.drop(columns = ['categories'])

    for col_name in categories_column_names:
        df[col_name] = df[col_name].apply(lambda x: int(x.split('-')[1]))

    return df, categories_column_names


def clean_data(df1, categories_column_names):
    """
    input:
        df - data frame
        categories_column_names - list of column names that shoud be cleand
    """
    df = copy.deepcopy(df1)

    for col_name in categories_column_names:
        if df[col_name].nunique() < 2:
            print(f'Dropping column {col_name} - {df[col_name].unique()} - only one unique value\n')
            df = df.drop(columns = col_name)

        elif df[col_name].nunique() > 2:
            print(f"Column '{col_name}' - {df[col_name].unique()} have to many unique values - {df[col_name].nunique()} should have only 2 [0, 1]\n")
            error_df = df[~df[col_name].isin([0, 1])]

            if len(error_df) / len(df) < 0.1:
                print(f"Removing rows with unvalid values - {error_df[col_name].unique()} that are less then 10% of dataset\n")
                df = df[df[col_name].isin([0, 1])]
                df[col_name] = df[col_name].apply(lambda x: bool(x))
            
            elif len(error_df) / len(df) < 0.2:
                print(f'Replacing unvalid values - {error_df[col_name].unique()} with mode of column - {df[col_name].mode()} that populates less then 20% of column values\n')
                mode_value = df[col_name].mode()
                df[col_name] = df[col_name].apply(lambda x: x if x.isin([0, 1]) else mode_value)
                df[col_name] = df[col_name].apply(lambda x: bool(x))

            else:
                print(f"Dropping column '{col_name}' - {df[col_name].unique()} - more then 20% of unvalid values\n")
                df.drop(columns = [col_name], inplace = True)

        else:
            df[col_name] = df[col_name].apply(lambda x: bool(x))

    return df


def save_to_database(df, database_path):
    engine = create_engine(f'sqlite:///{database_path}')
    df.to_sql('disaster', engine, index=False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:
        messages_path, categories_path, datbase_path, = sys.argv[1:]

    else:
        messages_path = '..\\data\\messages.csv'
        categories_path = '..\\data\\categories.csv' 
        datbase_path = '..\\disaster.db'

    print('Loading messages data...\n')
    messages = pd.read_csv(messages_path)

    print('Loading categories data...\n')
    categories = pd.read_csv(categories_path)

    print('Merging messages and categories on id...\n')
    df = messages.merge(on = 'id', right = categories)

    print('Spliting vaules in categories column into seperate columns...\n')
    df, categories_column_names = split_categories_columns(df)

    print('Cleaning data from unvalid valies != [0, 1] and turning them to bool type values...\n')
    df = clean_data(df, categories_column_names)

    print('Deduplication...\n')
    df = df.drop_duplicates(subset = ['id'], keep = False)
    df = df.drop_duplicates(subset = ['message'], keep = False)

    print('Saving to database...\n')
    save_to_database(df, datbase_path)

    print('Success !!!\n')
    print(f'Output: {datbase_path}\n')

    print(df.head())


if __name__ == '__main__':
    main()
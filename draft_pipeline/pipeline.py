import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from qwlist.qwlist import QList


def run_pipeline():
    print('Loading additional data...')
    stopwords = load_stopwords('../data/stopwords_pl.txt')

    print('Loading dataframes...')
    data = load_data('../data/train.tsv', sep='\t')

    clean_df = data
    clean_df['sentence'] = data['sentence'].apply(remove_usernames)

    print('Train-Test split...')
    train_df, test_df = train_test_split(clean_df)
    print()

    for model in [CountVectorizer(stop_words=stopwords, max_features=1000), TfidfVectorizer(stop_words=stopwords, max_features=1000)]:
        print(f'=== {model.__class__.__name__} ===')
        print('\tfit...')
        model.fit(train_df['sentence'])

        print('\ttransform...')
        x_train = model.transform(train_df['sentence']).toarray()
        y_train = train_df['target'].values

        x_test = model.transform(test_df['sentence']).toarray()
        y_test = test_df['target'].values

        for cls in [SVC(), RandomForestClassifier()]:
            print()
            print(f'\t=== {cls.__class__.__name__} ===')
            print('\t\tfit...')
            cls.fit(x_train, y_train)
            y_pred = cls.predict(x_test)
            print('\t\tf1 socre...')
            score = f1_score(y_test, y_pred)
            print('\t\t', score, '\n')
        
        print()



def load_data(path: str, sep=';') -> pd.DataFrame:
    return pd.read_csv(path, sep=sep, encoding='utf-8')


def load_stopwords(path: str) -> list[str]:
    with open(path, 'r', encoding='utf-8') as file:
        stopwords = [line.strip() for line in file.readlines()]
    return stopwords


def remove_usernames(text: str):
    s = QList(text.split(' '))
    if '@anonymized_account' in s:
        return " ".join(s.filter(lambda x: x != "@anonymized_account"))
    return " ".join(s)


if __name__ == '__main__':
    run_pipeline()

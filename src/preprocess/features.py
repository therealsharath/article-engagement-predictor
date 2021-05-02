#!/usr/bin/python3
# features.py

import os
import pandas as pd


def delete_incomplete_columns(file, to_be_deleted=[], not_to_be_deleted=[], special=False):
    data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', file))

    for tbd in to_be_deleted:
        del data[tbd]

    if len(not_to_be_deleted) > 0:
        data = data.filter(not_to_be_deleted)

    for feature in data:
        data = data[data[feature].notnull()]

    if special:
        unwanted = ['Interactive Feature', 'Obituary (Obit)', 'briefing', 'Letter']
        data = data[~data['material'].isin(unwanted)]

    data.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', os.path.splitext(file)[0] + '-dropped.csv'), index=False)


def one_hot_encoding(articles_dataset):
    articles_df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', articles_dataset))

    # Run one-hot encoding on material column
    for article in articles_df.index:
        if articles_df.at[article, 'material'] == 'News':
            articles_df.at[article, 'material'] = 'material_news'
        elif articles_df.at[article, 'material'] == 'Op-Ed':
            articles_df.at[article, 'material'] = 'material_op-ed'
        else:
            articles_df.at[article, 'material'] = 'material_other'

    # Keep News, Op-Ed, rename all others to other
    articles_df = pd.concat([articles_df.drop(columns=['material']), pd.get_dummies(articles_df['material'])], axis=1)
    articles_df.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', articles_dataset), index=False)


def drop_standalone_comments(articles_dataset, comments_dataset):
    articles_df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', articles_dataset))
    comments_df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', comments_dataset))

    articles = set()
    for article in articles_df.index:
        articles.add(articles_df.at[article, 'uniqueID'])

    comments_to_drop = [comment for comment in comments_df.index if comments_df.at[comment, 'articleID'] not in articles]
    comments_df = comments_df.drop(index=comments_to_drop)
    comments_df.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', comments_dataset), index=False)


# main function
def drop_features():
    # calling the function on articles
    tbd = ['newsdesk', 'keywords', 'pub_date', 'section', 'subsection']  # param2 columns we want to delete
    delete_incomplete_columns('nyt-articles-2020.csv', to_be_deleted=tbd, special=True)

    # Run one-hot encoding on articles
    one_hot_encoding('nyt-articles-2020-dropped.csv')

    # calling the function on comments
    tbd = ['commentBody', 'articleID']  # param2 columns we want to delete
    delete_incomplete_columns('nyt-comments-2020.csv', not_to_be_deleted=tbd)

    # drop standalone comments
    drop_standalone_comments('nyt-articles-2020-dropped.csv', 'nyt-comments-2020-dropped.csv')

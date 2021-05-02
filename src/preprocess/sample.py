#!/usr/bin/python3
# sample.py

import os
import pandas as pd


def get_proportional_comments_sample(percentage=0.25, articles_dataset=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'nyt-articles-2020-dropped.csv'), comments_dataset=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'nyt-comments-2020-dropped.csv')):
    articles_df = pd.read_csv(articles_dataset)
    comments_df = pd.read_csv(comments_dataset)

    articles = {}
    comments = []
    for article in articles_df.index:
        articles[articles_df.at[article, 'uniqueID']] = article
        comments.append([])

    for comment in comments_df.index:
        comments[articles[comments_df.at[comment, 'articleID']]].append(comment)

    pd.concat([comments_df.iloc[comments[article]].sample(frac=percentage).reset_index(drop=True) for article in articles_df.index]).to_csv(os.path.splitext(comments_dataset)[0] + '-sample.csv', index=False)

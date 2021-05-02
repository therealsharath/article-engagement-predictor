#!/usr/bin/python3
# sentiment_col.py

import os
import numpy as np
import pandas as pd
from transformers import pipeline
from sklearn.preprocessing import StandardScaler


def generate_sentiment_col(articles_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'nyt-articles-2020-dropped-cleaned.csv'), comments_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'nyt-comments-2020-dropped-sample-cleaned.csv')):
    articles_df = pd.read_csv(articles_path)
    comments_df = pd.read_csv(comments_path)
    nlp = pipeline('sentiment-analysis', device=0)

    # Run NLP sentiment analysis on each comment
    comments_df['sentiment_magnitude'] = 0
    batch_size = 100
    for batch in range(0, len(comments_df), batch_size):
        results = nlp(list(map(str, comments_df['commentBody'][batch:batch + batch_size].tolist())), truncation=True)
        print(f'NLP batch {batch} - {batch + batch_size} finished')
        sentiments = list(map(lambda x: abs(x['score']), results))
        comments_df['sentiment_magnitude'][batch:batch + batch_size] = sentiments
        if batch / batch_size % 100 == 0:
            comments_df.to_csv(os.path.splitext(comments_path)[0] + '-partial-sentiment.csv', index=False)
    comments_df.to_csv(os.path.splitext(comments_path)[0] + '-partial-sentiment.csv', index=False)

    # Scale sentiments
    scaler = StandardScaler()
    sentiments = np.array(comments_df['sentiment_magnitude'].tolist()).reshape((-1, 1))
    sentiments = scaler.fit_transform(sentiments)
    comments_df['sentiment_magnitude'] = sentiments.reshape((sentiments.shape[0],))

    articles = {}
    comments = []
    for article in articles_df.index:
        articles[articles_df.at[article, 'uniqueID']] = article
        comments.append([])

    for comment in comments_df.index:
        comments[articles[comments_df.at[comment, 'articleID']]].append(comment)

    for article in articles_df.index:
        article_comments = comments_df.iloc[comments[article]]['sentiment_magnitude'].tolist()
        average_article_comments_sentiment = (sum(article_comments) / len(article_comments)) if len(article_comments) > 0 else 0
        articles_df.at[article, 'avg_comment_sentiment_magnitude'] = average_article_comments_sentiment

    articles_df.to_csv(os.path.splitext(articles_path)[0] + '-sentiment.csv', index=False)
    comments_df.to_csv(os.path.splitext(comments_path)[0] + '-sentiment.csv', index=False)

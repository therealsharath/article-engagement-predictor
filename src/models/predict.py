#!/usr/bin/python3
# predict.py

import os
import pickle
import pandas as pd
from num_comments import generate_and_evaluate_model as generate_and_evaluate_n_comments_model
from sentiment_prediction import generate_and_evaluate_model as generate_and_evaluate_sentiment_prediction_model


def predict_engagement_score(dataset, random_sample=True):
    X = pd.read_csv(dataset)
    if random_sample:
        datapoint = X.sample(n=1)
    else:
        datapoint = X.iloc[0]
    print(datapoint)
    datapoint = datapoint.drop(columns=['n_comments'])

    # num_comments
    n_comments_model, n_comments_accuracy = generate_and_evaluate_n_comments_model(dataset, hyperparameterization=False, timeout=600)
    # n_comments_model = pickle.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'num_comments_model.pkl'), 'rb'))
    n_comments = n_comments_model.predict(datapoint)[0]
    print(f'n_comments: {n_comments}')

    # sentiment_prediction
    sentiment_prediction_model, sentiment_prediction_accuracy = generate_and_evaluate_sentiment_prediction_model(dataset, hyperparameterization=False, timeout=600)
    # sentiment_prediction_model = pickle.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'sentiment_prediction_model.pkl'), 'rb'))
    avg_comment_sentiment_magnitude = sentiment_prediction_model.predict(datapoint)[0]
    print(f'avg_comment_sentiment_magnitude: {avg_comment_sentiment_magnitude}')

    # Engagement metric
    return n_comments * avg_comment_sentiment_magnitude


if __name__ == '__main__':
    print('Engagement metric: ' + str(predict_engagement_score(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'nyt-articles-2020-final-dataset.csv'))))

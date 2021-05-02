#!/usr/bin/python3
# text.py

import os
import contractions
import pandas as pd
from word2number.w2n import word_to_num
from stop_words import get_stop_words


def clean_text(text):
    try:
        cleaned_text = contractions.fix(text.strip()).lower()
    except IndexError:
        cleaned_text = text.strip().lower()

    words = [x for x in cleaned_text.split(' ') if x]

    current_number = []
    i = 0
    while i < len(words):
        try:
            try:
                float(words[i])
                current_number.append(words[i])
                words.pop(i)
            except ValueError:
                pass

            if i < len(words):
                try:
                    word_to_num(words[i])
                except IndexError:
                    raise ValueError
                current_number.append(words[i])
                words.pop(i)
        except ValueError:
            if current_number:
                j = 0
                prod = 1
                while j < len(current_number):
                    try:
                        prod *= float(current_number[j])
                        current_number.pop(j)
                    except ValueError:
                        j += 1

                if current_number:
                    try:
                        num = prod * word_to_num(' '.join(current_number))
                    except:
                        num = prod
                else:
                    num = prod
                words.insert(i, str(int(num) if type(num) == float and num.is_integer() else num))
                current_number.clear()
                i += 1

            if len(words[i]) <= 2 or words[i] in get_stop_words('en'):
                words.pop(i)
                continue

            i += 1
    else:
        if current_number:
            try:
                words.append(str(word_to_num(' '.join(current_number))))
            except ValueError:
                words.append(' '.join(current_number))

    return ''.join(x for x in ' '.join(words) if x.isalnum() or x == ' ')


def preprocess_dataset_text():
    articles_file = 'nyt-articles-2020-dropped.csv'
    comments_file = 'nyt-comments-2020-dropped-sample.csv'
    articles_df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', articles_file))
    comments_df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', comments_file))

    for column in articles_df.columns:
        if articles_df[column].dtype == 'O' and column != 'uniqueID':
            articles_df[column] = articles_df[column].apply(clean_text)

    articles_df.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', os.path.splitext(articles_file)[0] + '-cleaned.csv'), index=False)
    del articles_df

    for column in comments_df.columns:
        if comments_df[column].dtype == 'O' and column != 'articleID':
            comments_df[column] = comments_df[column].apply(clean_text)

    comments_df.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', os.path.splitext(comments_file)[0] + '-cleaned.csv'), index=False)

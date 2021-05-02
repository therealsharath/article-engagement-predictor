from sentence_transformers import SentenceTransformer
import pandas as pd


def sentence_transform(articles_path='C:/Users/Nisha/Documents/CS4641/Project/nyt-articles-2020.csv/nyt-articles-2020.csv', comments_path='C:/Users/Nisha/Documents/CS4641/Project/nyt-comments-2020.csv/nyt-comments-2020.csv'):
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    articles = pd.read_csv(articles_path)
    comments = pd.read_csv(comments_path)
    article_comments = ""
    sentences = []
    for article_row in articles.index:
        article_id = articles.at[article_row, 'uniqueID']
        for comment_row in comments.index:
            if comments.at[comment_row, 'articleID'] == article_id:
                article_comments += comments.at[comment_row, 'commentBody']
        sentences.append(article_comments)

    sentence_embeddings = model.encode(sentences, normalize_embeddings=True)
    for sentence, embedding in zip(sentences, sentence_embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        print("")
        print(embedding.shape[0])


sentence_transform()

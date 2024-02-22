from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

docs = fetch_20newsgroups(subset='all')['data']
topic_model = BERTopic()
topics, probabilities = topic_model.fit_transform(docs)
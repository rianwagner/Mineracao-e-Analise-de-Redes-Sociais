import pandas as pd
import numpy as np
import networkx as nx
import json
from community import community_louvain
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

df = pd.read_csv('user_topics_sentiment.csv')
user_comments = df.groupby('user')['comment_text'].apply(
    lambda x: ' '.join(x.astype(str).fillna(''))
).reset_index()
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(user_comments['comment_text'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
threshold = 0.7
G = nx.Graph()

for i in range(len(cosine_sim)):
    G.add_node(i, user=user_comments.iloc[i]['user'])

for i in range(len(cosine_sim)):
    for j in range(i+1, len(cosine_sim)):
        if cosine_sim[i][j] > threshold:
            G.add_edge(i, j, weight=cosine_sim[i][j])

partition = community_louvain.best_partition(G)
user_comments['community'] = user_comments.index.map(partition)

df = pd.merge(df, user_comments[['user', 'community']], on='user', how='left')

community_stats = df.groupby('community').agg({
    'sentiment': 'mean',
    'topic': lambda x: x.mode()[0],
    'likes': 'mean',
    'number_responses': 'mean'
}).reset_index()

def recommend_content(user, n=10):
    user_community = user_comments[user_comments['user'] == user]['community'].values[0]
    community_comments = df[df['community'] == user_community]
    return community_comments.nlargest(n, 'likes')[['user', 'comment_text', 'likes']]

def plot_communities(G, partition):
    plt.figure(figsize=(15, 10))
    cmap = ListedColormap(plt.cm.tab20.colors)
    
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=50,
                           cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.1)
    plt.title('Comunidades de Usuários Detectadas', fontsize=14)
    plt.show()

plot_communities(G, partition)

topic_community = pd.crosstab(df['community'], df['topic'])
plt.figure(figsize=(10, 6))
sns.heatmap(topic_community, annot=True, cmap='Blues', fmt='d')
plt.title('Distribuição de Tópicos por Comunidade')
plt.xlabel('Tópico')
plt.ylabel('Comunidade')
plt.show()

print("\nExemplo de recomendação para usuario:", user_comments.iloc[0]['user'])
print(recommend_content(user_comments.iloc[0]['user']))
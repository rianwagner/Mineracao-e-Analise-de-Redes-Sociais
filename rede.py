import json
import re
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')
nltk.download('punkt')

with open('youtube_comments_EM', 'r', encoding='utf-8') as f:
    data = json.load(f)

def preprocess_text(text):
    text = text.lower() 
    text = re.sub(r'[^a-zA-ZÀ-ÿ0-9\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return filtered_tokens

word_lists = [preprocess_text(comment["comment_text"]) for comment in data["comments"]]

all_words = [word for words in word_lists for word in words]

word_freq = Counter(all_words)
top_words = [word for word, freq in word_freq.most_common(100)]

G = nx.Graph()
G.add_nodes_from(top_words)

for words in word_lists:
    filtered_words = [word for word in words if word in top_words]
    for word1, word2 in combinations(set(filtered_words), 2):
        if G.has_edge(word1, word2):
            G[word1][word2]['weight'] += 1
        else:
            G.add_edge(word1, word2, weight=1)

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
degree_dist = dict(G.degree())
clustering_coef = nx.average_clustering(G)

degree_centrality = nx.degree_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)

node_sizes = [5000 * eigenvector_centrality[node] for node in G.nodes()]

print(f"Número de nós: {num_nodes}")
print(f"Número de arestas: {num_edges}")
print(f"Coeficiente de clustering médio: {clustering_coef:.4f}")
print("Distribuição de graus (top 10):", sorted(degree_dist.items(), key=lambda x: x[1], reverse=True)[:10])
print("Centralidade de grau (top 10):", sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10])
print("Centralidade de eigenvector (top 10):", sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:10])

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=node_sizes, edge_color='gray', alpha=0.6, font_size=10)
plt.title(f"Rede de Coocorrência de Palavras\nNós: {num_nodes}, Arestas: {num_edges}, Clustering: {clustering_coef:.2f}")
plt.show()
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

with open('youtube_comments_EM', 'r', encoding='utf-8') as f:
    data = json.load(f)

comments = data['comments']
df = pd.DataFrame(comments)
numerical_stats = df[['likes', 'number_responses']].describe()
print("Estatísticas básicas (Atributos numéricos):\n", numerical_stats)
df['comment_length'] = df['comment_text'].apply(lambda x: len(x.split()))
length_stats = df['comment_length'].describe()
print("Estatísticas básicas (Comprimento do comentário):\n", length_stats)

comments = df['comment_text'].values 
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(comments)

scaler = StandardScaler(with_mean=False) 
X_scaled = scaler.fit_transform(X.toarray())

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X_scaled)

plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
plt.title('PCA - Redução de Dimensionalidade com Base em Correlação de Palavras')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(reduced_data)

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=df['cluster'], cmap='viridis')
plt.title('Clustering KMeans - Comentários com Base em Correlação de Palavras')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()

pd.DataFrame(reduced_data, columns=['PC1', 'PC2']).to_csv('pca_reduced_data.csv', index=False)
df[['user', 'cluster']].to_csv('user_clusters.csv', index=False)
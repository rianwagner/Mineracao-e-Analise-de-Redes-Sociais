import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

with open('youtube_comments_EM', 'r', encoding='utf-8') as f:
    data = json.load(f)

comments = data['comments'] 
df = pd.DataFrame(comments)
df['clean_comments'] = df['comment_text'].apply(preprocess_text)
print(df[['comment_text', 'clean_comments']].head())

vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df['clean_comments'])

analyzer = SentimentIntensityAnalyzer()
df['sentiment'] = df['comment_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

plt.figure(figsize=(8, 6))
sns.histplot(df['sentiment'], kde=True, color='blue', bins=30)
plt.title('Distribuição de sentimentos por comentários')
plt.xlabel('Valor de sentimento')
plt.ylabel('Frequência')
plt.show()

lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(X)

for index, topic in enumerate(lda.components_):
    print(f'Topic {index}:')
    print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
    print()

for index, topic in enumerate(lda.components_):
    plt.figure(figsize=(8, 6))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-50:]]))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Nuvem de palvras por tópico {index}')
    plt.axis('off')
    plt.show()

topic_values = lda.transform(X)
df['topic'] = topic_values.argmax(axis=1)

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='topic', palette='Set2')
plt.title('Didtribuição de comentários por tópicos')
plt.xlabel('Tópico')
plt.ylabel('Número de comentários')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='topic', y='sentiment', palette='Set2')
plt.title('Distribuição de sentimento por tópico')
plt.xlabel('Tópico')
plt.ylabel('Valor de sentimento')
plt.show()

df.to_csv('user_topics_sentiment.csv', index=False)

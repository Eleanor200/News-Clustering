import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# The following line must be the first Streamlit command used in your app, and it should only be set once.
st.set_page_config(page_title="Clustered News Articles from BBC", layout="wide")

def scrape_bbc_news():
    url = 'https://www.bbc.co.uk/news'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    article_tags = soup.find_all('a', {'class': 'gs-c-promo-heading'})

    articles = []
    for tag in article_tags[:20]:  # Limit to 20 articles for simplicity
        title = tag.text.strip()
        link = tag['href']
        if not link.startswith('http'):
            link = f'https://www.bbc.co.uk{link}'
        articles.append({'title': title, 'link': link})
   
    return articles

def cluster_articles(articles, n_clusters=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([article['title'] for article in articles])

    if X.shape[0] < n_clusters:
        n_clusters = X.shape[0]  # Ensure we do not have more clusters than samples
   
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
   
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
   
    clusters = {}
    for i in range(n_clusters):
        cluster_terms = [terms[ind] for ind in order_centroids[i, :10]]  # Top-10 terms per cluster
        clusters[i] = {'articles': [], 'terms': ', '.join(cluster_terms)}
       
    for i, label in enumerate(kmeans.labels_):
        clusters[label]['articles'].append(articles[i])
       
    return clusters

def show_clusters(n_clusters):
    articles = scrape_bbc_news()
    if articles:
        clusters = cluster_articles(articles, n_clusters=n_clusters)
        for cluster_id, cluster_info in clusters.items():
            st.subheader(f"Cluster {cluster_id + 1}")  # Add 1 to make cluster numbering start from 1
            st.markdown(f"**Top terms:** {cluster_info['terms']}")
            for article in cluster_info['articles']:
                st.markdown(f"[{article['title']}]({article['link']})", unsafe_allow_html=True)
    else:
        st.error("No articles found. Please check the URL or try a different site.")

# Streamlit UI
st.title('Clustered News Articles from BBC News')

# Sidebar for user input
st.sidebar.header("User Input Options")
n_clusters = st.sidebar.slider('Select number of clusters', min_value=2, max_value=20, value=5, step=1)

# Button to trigger scraping and clustering
if st.button('Scrape and Cluster Articles'):
    show_clusters(n_clusters)

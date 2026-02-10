import streamlit as st
import pandas as pd
import feedparser
import requests
from bs4 import BeautifulSoup
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models
from textblob import TextBlob
import spacy
import en_core_web_sm # <--- THIS IS THE FIRST CHANGE

# --- Page Configuration ---
st.set_page_config(
    page_title="Project Argus Prototype",
    page_icon="🛡️",
    layout="wide",
)

# --- NLTK and Spacy Setup ---
# Download necessary NLTK data if not present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# Load Spacy model using the direct import method
nlp = en_core_web_sm.load() # <--- THIS IS THE SECOND CHANGE

# --- Caching ---
# Cache data loading to avoid re-reading the file on every interaction
@st.cache_data
def load_sources():
    """Loads news sources from the CSV file."""
    return pd.read_csv("sources.csv")

# --- Core Functions ---

def display_source_filter(df):
    """Creates the sidebar UI for filtering sources and returns selected feed URLs."""
    st.sidebar.header("Source Filter")
    
    # Get unique countries and create a 'Select All' option
    countries = sorted(df['Country'].unique())
    
    # Use session state to manage checkbox states
    if 'all_countries' not in st.session_state:
        st.session_state.all_countries = True

    select_all_countries = st.sidebar.checkbox("Select All Countries", key='all_countries')

    selected_feeds = []
    
    for country in countries:
        country_df = df[df['Country'] == country]
        
        # Determine if a country checkbox should be checked
        if select_all_countries:
            st.session_state[f"country_{country}"] = True
        
        is_country_selected = st.sidebar.checkbox(country, key=f"country_{country}")
        
        if is_country_selected:
            for index, row in country_df.iterrows():
                # Determine if a source checkbox should be checked
                if select_all_countries or st.session_state[f"country_{country}"]:
                     st.session_state[f"source_{index}"] = True

                is_source_selected = st.sidebar.checkbox(f"  {row['Source Name']}", key=f"source_{index}", value=st.session_state.get(f"source_{index}", True))
                if is_source_selected:
                    selected_feeds.append(row['RSS Feed URL'])
                    
    return selected_feeds

@st.cache_data
def fetch_rss_feeds(feed_urls):
    """Fetches and parses articles from a list of RSS feed URLs."""
    articles = []
    for url in feed_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            articles.append({
                'title': entry.title,
                'summary': entry.get('summary', ''),
                'link': entry.link
            })
    return articles

@st.cache_data
def fetch_social_media_sample(keywords, limit=20):
    """Simulates fetching social media data by scraping Google search results for X.com."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    posts = []
    for keyword in keywords:
        query = f'site:x.com "{keyword}"'
        search_url = f"https://www.google.com/search?q={query}&num=5" # Fetch a few results per keyword
        try:
            response = requests.get(search_url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            # Find all text snippets in the search results
            snippets = soup.find_all('div', {'class': ['BNeawe', 's3v9rd', 'AP7Wnd']})
            for snippet in snippets:
                text = snippet.get_text()
                if len(text.split()) > 5: # Basic filter for meaningful text
                    posts.append({'text': text})
        except Exception as e:
            st.warning(f"Could not fetch social media for '{keyword}': {e}")
    return posts[:limit]


def preprocess_text(text):
    """Cleans and tokenizes text for analysis."""
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'<.*?>', '', text).lower() # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove punctuation
    tokens = nltk.word_tokenize(text)
    return [word for word in tokens if word not in stop_words and len(word) > 2]

def analyze_narratives(articles, num_topics=5, num_words=5):
    """Performs topic modeling and sentiment analysis."""
    if not articles:
        return [], 0.0

    # Topic Modeling (LDA)
    texts = [preprocess_text(article['title'] + " " + article.get('summary', article.get('text', ''))) for article in articles]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    if not corpus or not any(corpus):
        return [], 0.0

    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=100)
    topics = lda_model.print_topics(num_words=num_words)

    # Sentiment Analysis
    sentiment_score = sum(TextBlob(article['title']).sentiment.polarity for article in articles) / len(articles)
    
    return topics, sentiment_score

def get_keywords_from_articles(articles, num_keywords=10):
    """Extracts the most common named entities (organizations, people) to use as social media keywords."""
    text = " ".join([a['title'] for a in articles])
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ('GPE', 'ORG', 'PERSON')]
    return [item[0] for item in Counter(entities).most_common(num_keywords)]

# --- Streamlit App Layout ---
st.title("🛡️ Project Argus - Prototype v4")
st.markdown("An AI-powered dashboard for analyzing the global information landscape.")

# Load sources and display sidebar
sources_df = load_sources()
selected_feeds = display_source_filter(sources_df)

if not selected_feeds:
    st.warning("Please select at least one news source from the filter panel on the left.")
else:
    # --- Main Analysis Trigger ---
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        analyze_button = st.button("▶️ Run Global Analysis", use_container_width=True, type="primary")

    if analyze_button:
        # Fetch and Analyze News Data
        with st.spinner("Fetching and analyzing news articles..."):
            news_articles = fetch_rss_feeds(selected_feeds)
            news_topics, news_sentiment = analyze_narratives(news_articles)
        
        # Fetch and Analyze Social Media Data
        with st.spinner("Sampling and analyzing social media conversations..."):
            keywords_from_news = get_keywords_from_articles(news_articles)
            social_posts = fetch_social_media_sample(keywords_from_news)
            social_topics, social_sentiment = analyze_narratives(social_posts)

        st.success("Analysis Complete!")
        st.header("📊 Information Environment Dashboard")
        st.markdown("---")

        # --- Display Results ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📰 Top 5 News Narratives")
            if news_topics:
                sentiment_emoji = "😐"
                if news_sentiment > 0.1: sentiment_emoji = "🙂"
                if news_sentiment < -0.1: sentiment_emoji = "☹️"
                st.metric(label="Overall News Sentiment", value=f"{news_sentiment:.2f}", delta=sentiment_emoji)
                
                for i, topic in enumerate(news_topics):
                    st.info(f"**Narrative {i+1}:** `{', '.join(re.findall(r'\"(.*?)\"', topic[1]))}`")
            else:
                st.warning("Not enough data to identify news narratives.")
        
        with col2:
            st.subheader("📱 Top 5 Socials Narratives")
            if social_topics:
                sentiment_emoji = "😐"
                if social_sentiment > 0.1: sentiment_emoji = "🙂"
                if social_sentiment < -0.1: sentiment_emoji = "☹️"
                st.metric(label="Overall Socials Sentiment", value=f"{social_sentiment:.2f}", delta=sentiment_emoji)
                
                for i, topic in enumerate(social_topics):
                    st.info(f"**Narrative {i+1}:** `{', '.join(re.findall(r'\"(.*?)\"', topic[1]))}`")
            else:
                st.warning("Not enough data to identify social media narratives.")

        # --- Counter-Messaging Section ---
        st.markdown("---")
        st.header("✍️ Counter-Messaging Assistant")
        st.markdown("Use the insights from the dashboard to draft a prompt for an LLM.")
        
        prompt_template = f"""
        **Context:**
        - The news landscape is showing narratives about: {', '.join([re.findall(r'\"(.*?)\"', t[1])[0] for t in news_topics]) if news_topics else 'N/A'}.
        - Public social media conversation seems focused on: {', '.join([re.findall(r'\"(.*?)\"', t[1])[0] for t in social_topics]) if social_topics else 'N/A'}.

        **Task:**
        Draft [3] talking points for a [press briefing] to counter the narrative about...
        """
        st.text_area("LLM Prompt:", value=prompt_template, height=250)

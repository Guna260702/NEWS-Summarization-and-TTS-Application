import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Any, Optional
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
# from indic_tts_utils import generate_speech
import os
import json
from googlesearch import search
import time
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download necessary NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    logger.error(f"Failed to download NLTK resources: {e}")

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Initialize summarization pipeline
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception as e:
    logger.error(f"Failed to initialize summarization model: {e}")
    summarizer = None

def clean_text(text: str) -> str:
    """
    Clean and preprocess text
    """
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    return text

def fetch_news_articles(company_name: str, count: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch news articles about a company using Google search and web scraping
    """
    logger.info(f"Fetching news articles for {company_name}")
    
    articles = []
    query = f"{company_name} news"
    
    try:
        # Search for news articles
        search_results = search(query, num=20, stop=20, pause=2)
        urls = list(search_results)
        
        # Randomize to get a diverse set of sources
        random.shuffle(urls)
        
        for url in urls[:min(count*2, len(urls))]:  # Fetch more than needed in case some fail
            if len(articles) >= count:
                break
                
            # Skip websites that typically block scraping
            if any(domain in url for domain in ['bloomberg.com', 'forbes.com', 'wsj.com']):
                continue
                
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Get title
                    title_tag = soup.find('title')
                    if title_tag:
                        title = title_tag.text.strip()
                    else:
                        title = "No title found"
                    
                    # Get content - this is simplified; real implementation would need more nuance
                    paragraphs = soup.find_all('p')
                    content = ' '.join([p.text for p in paragraphs])
                    content = clean_text(content)
                    
                    if len(content) > 200:  # Only include articles with substantial content
                        articles.append({
                            "title": title,
                            "content": content,
                            "url": url
                        })
                        logger.info(f"Successfully scraped: {url}")
                    
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                continue
        
        logger.info(f"Successfully fetched {len(articles)} articles for {company_name}")
        return articles[:count]  # Return only up to the requested count
        
    except Exception as e:
        logger.error(f"Error fetching news articles: {e}")
        return []

def perform_sentiment_analysis(text: str) -> str:
    """
    Perform sentiment analysis on text and return sentiment category
    """
    try:
        sentiment_scores = sia.polarity_scores(text)
        compound_score = sentiment_scores['compound']
        
        if compound_score >= 0.05:
            return "Positive"
        elif compound_score <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return "Neutral"  # Default to neutral on error

def extract_topics(text: str, num_topics: int = 5) -> List[str]:
    """
    Extract main topics from text using keyword frequency analysis
    """
    try:
        # Tokenize and clean
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        
        # Filter out stopwords and non-alphabetic words
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 3]
        
        # Get frequency distribution
        fdist = FreqDist(filtered_words)
        
        # Return top keywords as topics
        topics = [word for word, _ in fdist.most_common(num_topics)]
        
        # Format topics with capitalization
        formatted_topics = [topic.capitalize() for topic in topics]
        
        return formatted_topics
    except Exception as e:
        logger.error(f"Error extracting topics: {e}")
        return ["General"]  # Default topic on error

def summarize_article(text: str, max_length: int = 150) -> str:
    """
    Generate a concise summary of the article text
    """
    try:
        # Use transformers summarization pipeline if available
        if summarizer:
            # Truncate input if it's too long
            input_text = text[:4096] if len(text) > 4096 else text
            
            summary = summarizer(input_text, max_length=max_length, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        else:
            # Fallback to extractive summarization
            sentences = sent_tokenize(text)
            
            if len(sentences) <= 3:
                return ' '.join(sentences)
            
            # Simple extractive summary - first 3 sentences
            return ' '.join(sentences[:3])
    except Exception as e:
        logger.error(f"Error summarizing article: {e}")
        # Fallback to returning a portion of the text
        return text[:200] + "..." if len(text) > 200 else text

def generate_comparative_analysis(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate comparative analysis across multiple articles
    """
    try:
        # Count sentiment distribution
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        for article in articles:
            sentiment_counts[article["sentiment"]] += 1
        
        # Identify common and unique topics
        all_topics = [set(article["topics"]) for article in articles]
        common_topics = set.intersection(*all_topics) if all_topics else set()
        
        # Get unique topics per article
        unique_topics = {}
        for i, article in enumerate(articles):
            article_name = f"Article {i+1}"
            other_topics = set().union(*[topics for j, topics in enumerate(all_topics) if j != i])
            unique_topics[article_name] = list(set(article["topics"]) - other_topics)
        
        # Generate comparisons between articles
        coverage_differences = []
        for i in range(len(articles) - 1):
            for j in range(i + 1, min(i + 2, len(articles))):  # Compare with next article
                comparison = {
                    "comparison": f"Article {i+1} focuses on {', '.join(articles[i]['topics'][:2])}, "
                                 f"while Article {j+1} discusses {', '.join(articles[j]['topics'][:2])}.",
                    "impact": generate_impact_statement(articles[i], articles[j])
                }
                coverage_differences.append(comparison)
        
        # Structure the complete comparative analysis
        comparative_analysis = {
            "sentiment_distribution": {
                "positive": sentiment_counts["Positive"],
                "negative": sentiment_counts["Negative"],
                "neutral": sentiment_counts["Neutral"]
            },
            "coverage_differences": coverage_differences[:5],  # Limit to 5 comparisons
            "topic_overlap": {
                "common_topics": list(common_topics),
                "unique_topics": unique_topics
            }
        }
        
        return comparative_analysis
    except Exception as e:
        logger.error(f"Error in comparative analysis: {e}")
        return {
            "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
            "coverage_differences": [],
            "topic_overlap": {"common_topics": [], "unique_topics": {}}
        }

def generate_impact_statement(article1: Dict[str, Any], article2: Dict[str, Any]) -> str:
    """
    Generate an impact statement comparing two articles
    """
    sentiment1 = article1["sentiment"]
    sentiment2 = article2["sentiment"]
    
    if sentiment1 == sentiment2:
        if sentiment1 == "Positive":
            return "Both articles present positive perspectives, reinforcing confidence in the company."
        elif sentiment1 == "Negative":
            return "Multiple negative reports may significantly impact investor sentiment."
        else:
            return "The neutral coverage across sources suggests steady performance without major developments."
    else:
        if (sentiment1 == "Positive" and sentiment2 == "Negative") or (sentiment1 == "Negative" and sentiment2 == "Positive"):
            return "Conflicting sentiments across sources may create market uncertainty."
        elif sentiment1 == "Neutral" or sentiment2 == "Neutral":
            return "Mixed signals with some balanced reporting may moderate overall market reaction."
    
    return "The differences in reporting may influence how investors perceive the company."


def generate_hindi_speech(text: str) -> str:
    """
    Convert text to Hindi speech and save as audio file
    """
    try:
        # Simple integration with Indic TTS
        # For actual implementation, you would use a proper TTS library like gTTS or an Indic TTS model
        audio_dir = "audio_files"
        os.makedirs(audio_dir, exist_ok=True)
        
        # Generate a unique filename
        timestamp = int(time.time())
        filename = f"{audio_dir}/speech_{timestamp}.mp3"
        
        # In a real implementation, this would convert text to speech
        # For now, we'll just log that it would happen
        logger.info(f"Would generate Hindi TTS for: {text[:50]}...")
        logger.info(f"Would save to file: {filename}")
        
        # Creating an empty file as a placeholder
        with open(filename, 'w') as f:
            f.write("")
        
        return filename
    except Exception as e:
        logger.error(f"Error generating Hindi speech: {e}")
        return ""
import streamlit as st
import requests
import json
import base64
import pandas as pd
import os
from logger import logging as lg
import time
from typing import Dict, Any, List

# Define the API URL
API_URL = "http://localhost:8000"  # Update this when deployed

def fetch_and_analyze(company_name: str, article_count: int = 10) -> Dict[str, Any]:
    """
    Send request to API to fetch and analyze news articles
    """
    try:
        response = requests.post(
            f"{API_URL}/analyze",
            json={"company_name": company_name, "article_count": article_count},
            timeout=500
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None

def generate_speech(text: str) -> str:
    """
    Send request to API to generate Hindi speech
    """
    try:
        response = requests.post(
            f"{API_URL}/generate-speech",
            json={"text": text},
            timeout=500
        )
        
        if response.status_code == 200:
            return response.json().get("audio_path", "")
        else:
            st.error(f"Error generating speech: {response.status_code} - {response.text}")
            return ""
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return ""
    
def generate_summary_speech(text: str, article_index: int) -> str:
    """
    Generate Hindi speech for an article summary and store the file path.
    """
    try:
        response = requests.post(
            f"{API_URL}/generate-speech",
            json={"text": text},
            timeout=60
        )
        if response.status_code == 200:
            audio_path = response.json().get("audio_path", "")
            st.session_state[f"audio_{article_index}"] = audio_path  # Store in session
            return audio_path
        else:
            st.error(f"Error generating speech: {response.status_code} - {response.text}")
            return ""
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return ""

def display_sentiment_distribution(sentiment_data: Dict[str, int]):
    """
    Display sentiment distribution as a bar chart
    """
    labels = ["Positive", "Negative", "Neutral"]
    values = [sentiment_data["positive"], sentiment_data["negative"], sentiment_data["neutral"]]
    
    df = pd.DataFrame({
        "Sentiment": labels,
        "Count": values
    })
    
    st.bar_chart(df.set_index("Sentiment"))

def display_articles(articles):
    """
    Display each article with an expandable section and a TTS button.
    """
    for i, article in enumerate(articles):
        with st.expander(f"{i+1}. {article['title']}"):
            st.write("**Summary:**", article["summary"])
            st.write("**Sentiment:**", article["sentiment"])
            st.write("**Topics:**", ", ".join(article["topics"]))
            st.write("**Source:**", article["url"])

            # Initialize session state for each article
            if f"audio_{i}" not in st.session_state:
                st.session_state[f"audio_{i}"] = None

            with st.spinner("Generating Hindi speech..."):
                audio_path = generate_summary_speech(article["summary"], i)
                if audio_path:
                        st.audio(audio_path, format="audio/mp3")
                else:
                    st.warning("Hindi speech generation failed or is not available in the demo.")


def display_comparative_analysis(analysis: Dict[str, Any]):
    """
    Display the comparative analysis section
    """
    st.subheader("Comparative Analysis")
    
    # Display sentiment distribution
    st.write("**Sentiment Distribution:**")
    display_sentiment_distribution(analysis["sentiment_distribution"])
    
    # Display coverage differences
    st.write("**Coverage Differences:**")
    for item in analysis["coverage_differences"]:
        st.write(f"- {item['comparison']}")
        st.write(f"  Impact: {item['impact']}")
    
    # Display topic overlap
    st.write("**Topic Analysis:**")
    st.write(f"- Common Topics: {', '.join(analysis['topic_overlap']['common_topics']) if analysis['topic_overlap']['common_topics'] else 'None'}")
    
    for article, topics in analysis["topic_overlap"]["unique_topics"].items():
        if topics:
            st.write(f"- Unique to {article}: {', '.join(topics)}")

def main():
    st.set_page_config(
        page_title="News Sentiment Analyzer",
        page_icon="📰",
        layout="wide"
    )
    
    st.title("📰 News Sentiment Analyzer with Hindi TTS")
    st.write("Enter a company name to analyze recent news sentiment and get a speech summary in Hindi.")
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        company_name = st.text_input("Company Name", "")
    
    with col2:
        article_count = st.number_input("Number of Articles", min_value=3, max_value=15, value=10)
    
    # Add some popular company options for quick selection
    sample_companies = ["Apple", "Tesla", "Google", "Microsoft", "Amazon", "Reliance Industries", "Tata Motors"]
    selected_sample = st.selectbox("Or choose a popular company:", [""] + sample_companies)
    
    if selected_sample:
        company_name = selected_sample
    
    analyze_button = st.button("Analyze News")
    
    if analyze_button and company_name:
        with st.spinner(f"Analyzing news for {company_name}..."):
            # Show a progress bar
            progress_bar = st.progress(0)
            for i in range(100):
                # Simulate progress
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Fetch and analyze news
            result = fetch_and_analyze(company_name, article_count)
            lg.info(result)
            
            if result:
                st.success("Analysis complete!")
                
                # Display final sentiment analysis
                st.header(f"Analysis for {result['company']}")
                st.subheader("Overall Sentiment")
                st.write(result["final_sentiment_analysis"])
                
                # Generate Hindi speech for the final analysis
                with st.spinner("Generating Hindi speech..."):
                    audio_path = generate_speech(result["final_sentiment_analysis"])
                    
                    if audio_path:
                        st.audio(audio_path, format="audio/mp3")
                    else:
                        st.warning("Hindi speech generation failed or is not available in the demo.")
                
                # Display tabs for different sections
                tab1, tab2 = st.tabs(["Articles", "Comparative Analysis"])
                
                with tab1:
                    display_articles(result["articles"])
                
                with tab2:
                    display_comparative_analysis(result["comparative_sentiment_score"])
    
    # Footer
    st.markdown("---")
    st.markdown("Built for the Akaike Internship Assignment | News Summarization and Text-to-Speech Application")

if __name__ == "__main__":
    main()
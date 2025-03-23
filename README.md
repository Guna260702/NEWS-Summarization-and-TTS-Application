# News Sentiment Analyzer with Hindi TTS

This web application extracts key details from multiple news articles related to a given company, performs sentiment analysis, conducts a comparative analysis, and generates a text-to-speech (TTS) output in Hindi. The tool allows users to input a company name and receive a structured sentiment report along with an audio output.

## Features

- **News Extraction**: Extracts and displays the title, summary, and other relevant metadata from news articles related to the given company using BeautifulSoup.
- **Sentiment Analysis**: Performs sentiment analysis on the article content (positive, negative, neutral).
- **Comparative Analysis**: Conducts a comparative sentiment analysis across articles to derive insights on how the company's news coverage varies.
- **Text-to-Speech**: Converts the summarized content into Hindi speech using an open-source TTS model.
- **User Interface**: Provides a simple web-based interface using Streamlit.
- **API Development**: Communication between the frontend and backend happens via APIs built with FastAPI.

## Architecture

The application follows a client-server architecture:

1. **Frontend**: Streamlit web application that provides the user interface
2. **Backend API**: FastAPI service that handles the core functionality
3. **Utility Functions**: Handles news extraction, sentiment analysis, and TTS generation

## Project Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Guna260702/NEWS-Summarization-and-TTS-Application.git
cd NEWS Summarization and TTS Application
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK resources:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
```

### Running the Application

1. Start the API server:
```bash
uvicorn api:app --reload
```

2. In a separate terminal, start the Streamlit application:
```bash
streamlit run app.py
```

3. Open your browser and navigate to `http://localhost:8501` to access the application.

## API Documentation

The application exposes the following API endpoints:

### `/analyze` (POST)

Analyzes news articles for a given company.

**Request Body**:
```json
{
  "company_name": "Tesla",
  "article_count": 10
}
```

**Response**:
```json
{
  "company": "Tesla",
  "articles": [
    {
      "title": "Tesla's Q3 Earnings Beat Expectations",
      "summary": "Tesla reported strong Q3 earnings...",
      "sentiment": "Positive",
      "topics": ["Earnings", "Growth", "Electric Vehicles"],
      "url": "https://example.com/article1"
    },
    ...
  ],
  "comparative_sentiment_score": {
    "sentiment_distribution": {
      "positive": 7,
      "negative": 2,
      "neutral": 1
    },
    "coverage_differences": [...],
    "topic_overlap": {...}
  },
  "final_sentiment_analysis": "Tesla's latest news coverage is mostly positive. Potential stock growth expected."
}
```

### `/generate-speech` (POST)

Generates Hindi speech from text.

**Request Body**:
```json
{
  "text": "Tesla's latest news coverage is mostly positive. Potential stock growth expected."
}
```

**Response**:
```json
{
  "audio_path": "/path/to/audio/file.mp3"
}
```

## Models Used

- **Sentiment Analysis**: Uses NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment analysis
- **Summarization**: Uses Facebook's BART-large-CNN model for article summarization
- **Text-to-Speech**: Uses gTTS (Google Text-to-Speech) for Hindi speech generation

## Limitations and Assumptions

- The application may not be able to scrape content from websites that block web scraping
- Sentiment analysis is performed using a pre-trained model and may not capture domain-specific sentiment nuances
- The application assumes that news articles are in English and translates to Hindi for TTS
- Topic extraction is based on keyword frequency and may not always capture the most relevant topics

## Deployment

The application is deployed on Hugging Face Spaces and can be accessed at:
[https://huggingface.co/spaces/yourusername/news-sentiment-analyzer](https://huggingface.co/spaces/yourusername/news-sentiment-analyzer)

## Future Improvements

- Implement more sophisticated news extraction techniques
- Add support for more languages
- Improve topic extraction using more advanced NLP techniques
- Add caching to improve performance

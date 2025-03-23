from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class TextToSpeechRequest(BaseModel):
    text: str

class CompanyRequest(BaseModel):
    company_name: str
    article_count: int = 10

class ArticleResponse(BaseModel):
    title: str
    summary: str
    sentiment: str
    topics: List[str]
    url: str

class ComparisonItem(BaseModel):
    comparison: str
    impact: str

class TopicOverlap(BaseModel):
    common_topics: List[str]
    unique_topics: Dict[str, List[str]]

class SentimentDistribution(BaseModel):
    positive: int
    negative: int
    neutral: int

class ComparativeSentiment(BaseModel):
    sentiment_distribution: SentimentDistribution
    coverage_differences: List[ComparisonItem]
    topic_overlap: TopicOverlap

class NewsResponse(BaseModel):
    company: str
    articles: List[ArticleResponse]
    comparative_sentiment_score: ComparativeSentiment
    final_sentiment_analysis: str
    audio_path: Optional[str] = None